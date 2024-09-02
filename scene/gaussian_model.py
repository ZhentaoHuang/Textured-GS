#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH, SH2RGB, SH2OPA, OPA2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._texture_opacity = torch.empty(0)
        # self._texture = torch.empty(0)  # Texture params
        self._texture_dc = torch.empty(0)
        self._texture_rest = torch.empty(0)
        self.pixel_count = torch.empty(0)
        self.sig_out = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 1
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_texture(self):
        texture_dc = self._texture_dc
        texture_rest = self._texture_rest
        return torch.cat((texture_dc, texture_rest), dim=1)
    
    @property
    def get_texture_opacity(self):
        return self._texture_opacity
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1



    def create_from_pcd_masked(self, pcd: BasicPointCloud, spatial_lr_scale: float, reference_index: int=2100, num_neighbors: int = 1):
        self.spatial_lr_scale = spatial_lr_scale
        # Load and convert point cloud data to PyTorch tensors on CUDA
        fused_point_cloud = torch.tensor(np.asarray(pcd.points), dtype=torch.float).cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors), dtype=torch.float).cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2), dtype=torch.float).cuda()
        # features[:, :3, 0] = fused_color
        # features[:, :3, 1:] = 0.0

        print("Number of points at initialisation: ", fused_point_cloud.shape[0])

        # Compute pairwise distances from each point to the reference point
        ref_point = fused_point_cloud[reference_index].unsqueeze(0)  # Reference point
        distances = torch.norm(fused_point_cloud - ref_point, dim=1)
        nearest_indices = torch.argsort(distances)[:num_neighbors]  # Indices of the nearest neighbors

        # Filter data to include only the 50 nearest neighbors
        fused_point_cloud = fused_point_cloud[nearest_indices]
        fused_color = fused_color[nearest_indices]
        features = features[nearest_indices]

        # Calculate scales based on distances, keep z dimension fixed
        dist2 = torch.clamp_min(torch.norm(fused_point_cloud - ref_point, dim=1, keepdim=True), 0.0000001)
        scales = torch.log(torch.sqrt(dist2)).repeat(1, 3)
        scales[:, 2] = -100  # Setting the z dimension scale fixed

        # Set rotations and opacities
        rots = torch.zeros((num_neighbors, 4), device="cuda")
        rots[:, 0] = 1  # Default rotation quaternion
        opacities = inverse_sigmoid(0.1 * torch.ones((num_neighbors, 1), dtype=torch.float, device="cuda"))

        # Update PyTorch tensors to Parameters
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(False))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(False))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


        # Update texture and pixel count parameters
        self._texture = nn.Parameter(features.clone().detach().transpose(1, 2).contiguous().to('cuda')).requires_grad_(True)
        self.pixel_count = torch.zeros((num_neighbors,), device="cuda").requires_grad_(False)

    

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, :3, 1:] = 0.0

        print("Number of points at initialisation in create_from_pcd: ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        # set z dimension fixed
        # scales[:,2] = -100
        # print(scales)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        # self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(False))
        # self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(False))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        # self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        texture_opacity = torch.zeros((self.get_xyz.shape[0], 16)) 
        texture_opacity[:,0] = texture_opacity[:,0] * -7.788
            # no detaching???

            # texture_opacity[:,0] = torch.from_numpy(opacities.copy()).squeeze(1)
            # texture_opacity[:,0] = texture_opacity[:,0] / 0.28209479177387814
        # texture_opacity[:, 1:] = 0.0

        # t = np.array([[0, 1, 2], [2, 5, 6], [1, 1, 1], [0.5, 0.5, 0.5]], dtype=np.float32)
        # t = np.array([[0.00001, 0.00001, 0.00001], [0.00001, 0.00001, 0.00001], [1, 1, 1], [0.5, 0.5, 0.5],[0.5, 0.5, 0.5],[0.00001, 0.00001, 0.00001], [0.00001, 0.00001, 0.00001], [1, 1, 1], [0.5, 0.5, 0.5]], dtype=np.float32)
        # t = np.random.rand(9, 3).astype(np.float32)
        # tensor_np = np.tile(t, (self._opacity.shape[0], 1, 1))
        # tensor_np = features.clone().detach().transpose(1,2)
        # # Convert the numpy array to a PyTorch tensor
        # self._texture = nn.Parameter(torch.tensor(tensor_np, device="cuda").requires_grad_(True))
## TODO direct clone?
        texture_dc = features[:,:,0:1].clone().detach()
        self.active_sh_degree = 0
        # Create a Parameter and ensure it requires gradients
        # self._texture = nn.Parameter(tensor_transposed).requires_grad_(True)
        # texture_dc = fused_color.clone().detach()
        texture_rest = torch.zeros((self.get_xyz.shape[0], 3, 15))
        self._texture_dc = nn.Parameter(texture_dc.clone().detach().requires_grad_(True).transpose(1, 2).contiguous().float().to(device="cuda"))
        self._texture_rest = nn.Parameter(texture_rest.clone().detach().requires_grad_(True).transpose(1, 2).contiguous().float().to(device="cuda"))
        # self._texture_opacity = nn.Parameter(texture_opacity.requires_grad_(True))
        self._texture_opacity = nn.Parameter(texture_opacity.clone().detach().requires_grad_(True).contiguous().float().to(device="cuda"))

        self.sig_out = torch.zeros((self.get_xyz.shape[0], 3), device='cuda').requires_grad_(False)
        self.pixel_count = torch.zeros((self.get_xyz.shape[0]), device="cuda").requires_grad_(False)

        # print(self._scaling, self._xyz)



    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # print("scaling", self._scaling.shape)


        # print(self._scaling.is_leaf)  # Should return True
        # self._scaling.requires_grad_(True)  # Enable gradients for all elements
        # self._scaling[:, 2].detach()  # Disable gradients for the fixed dimension




        # print("self.spatial_lr_scale", self.spatial_lr_scale, training_args.position_lr_init)
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            # {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            # {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            # {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            # {'params': [self._texture], 'lr': training_args.feature_lr, "name": "texture"}
            {'params': [self._texture_dc], 'lr': training_args.texture_lr_init, "name": "texture_dc"},
            {'params': [self._texture_rest], 'lr': training_args.texture_lr_init, "name": "texture_rest"},
            {'params': [self._texture_opacity], 'lr': training_args.opacity_lr, "name": "texture_opacity"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.texture_scheduler_args = get_expon_lr_func(lr_init=training_args.texture_lr_init,
                                                    lr_final=training_args.texture_lr_final,
                                                    lr_delay_mult=training_args.texture_lr_delay_mult,
                                                    max_steps=training_args.texture_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                
                # if iteration >= 14000:
                #     lr = 0
                #     param_group['lr'] = lr
                # else:
                #     lr = self.xyz_scheduler_args(iteration)
                #     param_group['lr'] = lr
                
                # return lr
            elif param_group["name"] == "texture_dc":
                if iteration >= 14000:

                    lr = self.texture_scheduler_args(iteration)
                    param_group['lr'] = lr
                else:
                    lr = self.texture_scheduler_args(iteration)
                    param_group['lr'] = lr

            elif param_group["name"] == "texture_rest":
                lr = self.texture_scheduler_args(iteration)/1.0
                param_group['lr'] = lr
            # elif param_group["name"] == "scaling":
            #     if iteration == 14000:
            #         lr = 0
            #         param_group['lr'] = lr
            # elif param_group["name"] == "rotation":
            #     if iteration == 14000:
            #         lr = 0
            #         param_group['lr'] = lr
            # elif param_group["name"] == "opacity":
            #     if iteration == 14000:
            #         lr = 0
            #         param_group['lr'] = lr


    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        # for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
        #     l.append('f_dc_{}'.format(i))
        # for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
        #     l.append('f_rest_{}'.format(i))
        # l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._texture_dc.shape[1]*self._texture_dc.shape[2]):
            l.append('text_dc_{}'.format(i))
        for i in range(self._texture_rest.shape[1]*self._texture_rest.shape[2]):
            l.append('text_rest_{}'.format(i))
        for i in range(self._texture_opacity.shape[1]):
            l.append('text_opacity_{}'.format(i))
        return l
    



    def save_ply_maksed(self, path, filter_opacity_threshold=None, region_bounds=None):
        mkdir_p(os.path.dirname(path))

        # Detach and transfer data from GPU to CPU memory
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)  # Assuming no normals are calculated
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scales = self._scaling.detach().cpu().numpy()
        rotations = self._rotation.detach().cpu().numpy()
        texture = self._texture.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

        # Apply filters
        mask = np.ones(len(xyz), dtype=bool)  # Initialize mask to include all Gaussians
        if filter_opacity_threshold is not None:
            mask &= (opacities > filter_opacity_threshold).flatten()  # Filter by opacity
        if region_bounds is not None:
            xmin, xmax, ymin, ymax, zmin, zmax = region_bounds
            mask &= (xyz[:, 0] >= xmin) & (xyz[:, 0] <= xmax)
            mask &= (xyz[:, 1] >= ymin) & (xyz[:, 1] <= ymax)
            mask &= (xyz[:, 2] >= zmin) & (xyz[:, 2] <= zmax)

        # Filter the data
        filtered_xyz = xyz[mask]
        filtered_normals = normals[mask]
        filtered_f_dc = f_dc[mask]
        filtered_f_rest = f_rest[mask]
        filtered_opacities = opacities[mask]
        filtered_scales = scales[mask]
        filtered_rotations = rotations[mask]
        filtered_texture = texture[mask]

        # Combine all attributes
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(len(filtered_xyz), dtype=dtype_full)
        attributes = np.concatenate((
            filtered_xyz, filtered_normals, filtered_f_dc, filtered_f_rest,
            filtered_opacities, filtered_scales, filtered_rotations, filtered_texture
        ), axis=1)
        elements[:] = list(map(tuple, attributes))
        
        # Create a PlyElement object and write to file
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el], text=True).write(path)


    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz) # No normals
        # f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # opacities = self._opacity.detach().cpu().numpy()
        opacities = np.ones((xyz.shape[0], 1))
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        texture_dc = self._texture_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        texture_rest = self._texture_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        texture_opacity = self._texture_opacity.detach().cpu().numpy()


        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, scale, rotation, texture_dc, texture_rest, texture_opacity), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opa = SH2OPA(self.get_texture_opacity[:,0])

        opacities_new = OPA2SH(torch.minimum(opa, torch.ones_like(opa) * 0.01))
        
        # opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        new_texture_opacity = self.get_texture_opacity.clone().detach()
        new_texture_opacity[:,0] = opacities_new
        optimizable_tensors = self.replace_tensor_to_optimizer(new_texture_opacity, "texture_opacity")
        self._texture_opacity = optimizable_tensors["texture_opacity"]
        # print("reset_opacity: ", opa.min(), opa.max(), "new: ", opacities_new.min(), opacities_new.max(), self.get_texture_opacity)



    def load_ply_half(self, path):
        plydata = PlyData.read(path)

        # Calculate half of the number of data points
        half_point = len(plydata.elements[0]["x"]) // 1
        # half_point = 5000
        # half_point =1200

        # Load only half of the data points
        xyz = np.stack((
            np.asarray(plydata.elements[0]["x"][:half_point]),
            np.asarray(plydata.elements[0]["y"][:half_point]),
            np.asarray(plydata.elements[0]["z"][:half_point])
        ), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"][:half_point])[..., np.newaxis]

        # Prepare features_dc for only half of the data
        features_dc = np.zeros((half_point, 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"][:half_point])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"][:half_point])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"][:half_point])

        # Prepare features_rest for only half of the data
        # extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        # extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        # features_extra = np.zeros((half_point, len(extra_f_names)))
        # for idx, attr_name in enumerate(extra_f_names):
        #     features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name][:half_point])
        # features_extra = features_extra.reshape((half_point, 3, (self.max_sh_degree + 1) ** 2 - 1))

        # Assuming texture data exists and there are 48 texture properties
        text_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("text_")]
        text_names = sorted(text_names, key=lambda x: int(x.split('_')[-1]))
        features_dc = torch.from_numpy(features_dc)
        if (len(text_names) == 48):
            texture_dc = features_dc.clone().detach()
            texture_rest = torch.zeros((half_point, 3, 15))

        else:
            print("No texture loaded, generating zeros!")
            # texture = torch.zeros((half_point, 3, 16))
            
            # texture[:, :, 0] = features_dc.squeeze(-1)
            # texture[:,:,0] = np.clip(texture[:,:,0], 0.000001, 0.9999999)
            # texture[:,:,0] = -np.log((1 - 0.28209479177387814*texture[:,:,0]-0.5)/0.28209479177387814*texture[:,:,0]+0.5)
            texture_dc = features_dc.clone().detach()
            rgb = SH2RGB(texture_dc)

            # texture_dc = texture_dc.squeeze(-1)
            # texture_dc = np.clip(texture_dc, 0.01, 0.99)
            # texture_dc = -np.log((1 - texture_dc)/texture_dc)
            # texture_dc =  texture_dc * (1 / 0.28209479177387814)
            rgb = np.clip(rgb, 0.000001, 0.999999)
            # texture_dc = -np.log((1 - 0.28209479177387814*texture_dc)/0.28209479177387814*texture_dc)
            # texture_dc = -np.log((1 - rgb)/rgb)
            texture_dc = OPA2SH(rgb)
            # texture_dc = texture_dc / 0.28209479177387814
            # texture_dc = 1 / (1 + np.exp(-(0.28209479177387814 * texture_dc + 0.5)))
            # texture_dc = -1/0.281 * np.log(1/(0.281 * texture_dc + 0.5) - 1)
            texture_rest = torch.zeros((half_point, 3, 15))
            texture_opacity = torch.zeros((half_point, 16)) 
            texture_opacity[:,0] = torch.from_numpy(opacities.copy()).squeeze(1)

            
            # no detaching???

            # texture_opacity[:,0] = torch.from_numpy(opacities.copy()).squeeze(1)
            texture_opacity[:,0] = texture_opacity[:,0] / 0.28209479177387814
            texture_opacity[:, 1:] = 0.0

          

        # Create a Parameter and ensure it requires gradients
            # self._texture = nn.Parameter(tensor_transposed).requires_grad_(True)
            
    
            # self._texture = nn.Parameter(torch.tensor(texture, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
            # texture = np.zeros((half_point, len(text_names)))

        # texture = np.zeros((half_point, len(text_names)))
        # for idx, attr_name in enumerate(text_names):
        #     texture[:, idx] = np.asarray(plydata.elements[0][attr_name][:half_point])
        # texture = texture.reshape((half_point, 3, 16))

        # Prepare scales and rotations for only half of the data
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((half_point, len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name][:half_point])
        # scales = torch.from_numpy(scales)
        # scales[:,2] = -100
        # scales[:,2] = scales[:,0]*1

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((half_point, len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name][:half_point])
        # rots = torch.from_numpy(rots)

        # dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(xyz)).float().cuda()), 0.0000001)
        
        # scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        # # # set z dimension fixed
        # # # scales[:,2] = -100
        # # # print(scales)
        # rots = torch.zeros((half_point, 4), device="cuda")
        # rots[:, 0] = 1
        

        # Set tensor parameters
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        # self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        # self._features_dc = nn.Parameter(features_dc.clone().detach().transpose(1, 2).contiguous().float().to(device="cuda"))

        # self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        # self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        # self._scaling = nn.Parameter(scales.clone().detach().requires_grad_(True).contiguous().float().to(device="cuda"))
        # self._rotation = nn.Parameter(rots.clone().detach().requires_grad_(True).contiguous().float().to(device="cuda"))
        # self._texture = nn.Parameter(torch.tensor(texture, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        # self._texture_dc = nn.Parameter(torch.tensor(texture_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._texture_dc = nn.Parameter(texture_dc.clone().detach().requires_grad_(True).transpose(1, 2).contiguous().float().to(device="cuda"))
        # self._texture_rest = nn.Parameter(torch.tensor(texture_rest, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._texture_rest = nn.Parameter(texture_rest.clone().detach().requires_grad_(True).transpose(1, 2).contiguous().float().to(device="cuda"))
        self._texture_opacity = nn.Parameter(texture_opacity.clone().detach().requires_grad_(True).contiguous().float().to(device="cuda"))
        self.max_radii2D = torch.zeros((half_point), device="cuda")
        self.active_sh_degree = 0
        self.pixel_count = torch.zeros((half_point), device="cuda").requires_grad_(False)
        self.sig_out = torch.zeros((half_point, 3), device='cuda').requires_grad_(False)


    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        # opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        # features_dc = np.zeros((xyz.shape[0], 3, 1))
        # features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        # features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        # features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        # extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        # assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        # features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        # for idx, attr_name in enumerate(extra_f_names):
        #     features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        # features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))


        # text_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("text_")]
        # text_names = sorted(text_names, key = lambda x: int(x.split('_')[-1]))
        # assert len(text_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        # assert len(text_names)==48
        extra_text_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("text_rest_")]
        extra_text_names = sorted(extra_text_names, key = lambda x: int(x.split('_')[-1]))
        # features_dc = torch.from_numpy(features_dc)
        if (len(extra_text_names) == 45):
            # print(extra_text_names)
            texture_extra = np.zeros((xyz.shape[0], len(extra_text_names)))
            for idx, attr_name in enumerate(extra_text_names):
                texture_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            texture_extra = texture_extra.reshape((texture_extra.shape[0], 3, 15))
            texture_extra = torch.from_numpy(texture_extra).clone().detach()

            texture_dc = np.zeros((xyz.shape[0], 3, 1))
            texture_dc[:, 0, 0] = np.asarray(plydata.elements[0]["text_dc_0"])
            texture_dc[:, 1, 0] = np.asarray(plydata.elements[0]["text_dc_1"])
            texture_dc[:, 2, 0] = np.asarray(plydata.elements[0]["text_dc_2"])
            texture_dc = torch.from_numpy(texture_dc).clone().detach()
            # texture_opacity = np.ones((xyz.shape[0], 16))
            # texture_opacity[:, 0] = opacities
            # texture_opacity[:, 1:] = 0.0

            # texture_opacity = torch.from_numpy(texture_opacity).clone().detach()
            
        else:
            print("No texture loaded, generating zeros!")
            
            texture_dc = features_dc.clone().detach()
            # texture_dc = features_dc.copy()
            # texture_dc = np.maximum(0, texture_dc)
            # # texture_dc += 0.5
            # texture_dc = np.clip(texture_dc, 0.01, 0.99)
            # texture_dc = -np.log((1 - texture_dc)/texture_dc)
            texture_extra = torch.zeros((xyz.shape[0], 3, 15))
            # Determine the max value for scaling; this could be the max of texture_dc or a predefined max if known
            max_value = 1
            min_value = 0.5  # Since the minimum after ReLU and adding 0.5 is 0.5

            # Scale values to be within 0 and 1
            texture_dc =  texture_dc * (1 / 0.28209479177387814)
            
    
        
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # Set the z-axis to a constant
        # scales[:, 2] = -100
        # scales[:,2] = scales[:,0]
        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        texture_opacity_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("text_opacity_")]
        texture_opacity_names = sorted(texture_opacity_names, key = lambda x: int(x.split('_')[-1]))
        texture_opacity = np.zeros((xyz.shape[0], len(texture_opacity_names)))
        for idx, attr_name in enumerate(texture_opacity_names):
            texture_opacity[:, idx] = np.asarray(plydata.elements[0][attr_name])

        print("reading!: ", texture_opacity)

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        # self._features_dc = nn.Parameter(features_dc.clone().detach().transpose(1, 2).contiguous().float().to(device="cuda"))
        # self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        # self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        # self._texture = nn.Parameter(torch.tensor(texture, dtype=torch.float, device="cuda").transpose(2, 1).contiguous().requires_grad_(True))
        self._texture_dc = nn.Parameter(texture_dc.clone().detach().requires_grad_(True).transpose(1, 2).contiguous().float().to(device="cuda"))
        # self._texture_rest = nn.Parameter(torch.tensor(texture_rest, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._texture_rest = nn.Parameter(texture_extra.clone().detach().requires_grad_(True).transpose(1, 2).contiguous().float().to(device="cuda"))
        # self._texture_opacity = nn.Parameter(texture_opacity.clone().detach().requires_grad_(True).contiguous().float().to(device="cuda"))
        self._texture_opacity = nn.Parameter(torch.tensor(texture_opacity, dtype=torch.float, device="cuda").requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.active_sh_degree = self.max_sh_degree
        self.pixel_count = torch.zeros((self.get_xyz.shape[0]), device="cuda").requires_grad_(False)
        self.sig_out = torch.zeros((self.get_xyz.shape[0], 3), device='cuda').requires_grad_(False)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # tensor_name = getattr(group['params'][0], 'name', 'Unnamed Tensor')
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points_textured(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        # self._xyz = optimizable_tensors["xyz"]
        # self._features_dc = optimizable_tensors["f_dc"]
        # self._features_rest = optimizable_tensors["f_rest"]
        # self._opacity = optimizable_tensors["opacity"]
        # self._scaling = optimizable_tensors["scaling"]
        # self._rotation = optimizable_tensors["rotation"]


        self._xyz = optimizable_tensors["xyz"]
        self._texture_dc = optimizable_tensors["texture_dc"]
        self._texture_rest = optimizable_tensors["texture_rest"]
        self._texture_opacity = optimizable_tensors["texture_opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        self.pixel_count = self.pixel_count[valid_points_mask]
        self.sig_out = self.sig_out[valid_points_mask]


    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()



        # l = [
        #     {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
        #     # {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
        #     # {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
        #     # {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
        #     {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
        #     {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        #     # {'params': [self._texture], 'lr': training_args.feature_lr, "name": "texture"}
        #     {'params': [self._texture_dc], 'lr': training_args.texture_lr_init, "name": "texture_dc"},
        #     {'params': [self._texture_rest], 'lr': training_args.texture_lr_init / 10.0, "name": "texture_rest"},
        #     {'params': [self._texture_opacity], 'lr': training_args.texture_lr_init/2.0, "name": "texture_opacity"}
        # ]



    def densification_postfix_textured(self, new_xyz, new_texture_dc, new_texture_rest, new_texture_opacity, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "texture_dc": new_texture_dc,
        "texture_rest": new_texture_rest,
        "texture_opacity": new_texture_opacity}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._texture_dc = optimizable_tensors["texture_dc"]
        self._texture_rest = optimizable_tensors["texture_rest"]
        self._texture_opacity = optimizable_tensors["texture_opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.pixel_count = torch.zeros((self.get_xyz.shape[0]), device="cuda").requires_grad_(False)
        self.sig_out = torch.zeros((self.get_xyz.shape[0], 3), device='cuda').requires_grad_(False)




    def densify_and_clone_textured(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        # new_features_dc = self._features_dc[selected_pts_mask]
        # new_features_rest = self._features_rest[selected_pts_mask]
        # new_opacities = self._opacity[selected_pts_mask]
        new_texture_dc = self._texture_dc[selected_pts_mask]
        new_texture_rest = self._texture_rest[selected_pts_mask]
        new_texture_opacity = self._texture_opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix_textured(new_xyz, new_texture_dc, new_texture_rest, new_texture_opacity, new_scaling, new_rotation)



    def densify_and_split_textured(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_texture_dc = self._texture_dc[selected_pts_mask].repeat(N,1,1)
        new_texture_rest = self._texture_rest[selected_pts_mask].repeat(N,1,1)
        new_texture_opacity = self._texture_opacity[selected_pts_mask].repeat(N,1)

        # self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        self.densification_postfix_textured(new_xyz, new_texture_dc, new_texture_rest, new_texture_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))

        self.prune_points_textured(prune_filter)


    def densify_and_prune_textured(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        if self.get_xyz.shape[0] <= 5000000:
            self.densify_and_clone_textured(grads, max_grad, extent)
            self.densify_and_split_textured(grads, max_grad, extent)
        

        opa = SH2OPA(self.get_texture_opacity[:,0])
        # opa = torch.sigmoid(opa)
        prune_mask = (opa < min_opacity).squeeze()
        # prune_mask = torch.zeros(self.get_xyz.shape[0],  dtype=torch.bool)
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points_textured(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1