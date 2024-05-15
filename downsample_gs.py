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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

import numpy as np

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        print("view", view)
        print("pipeline", pipeline)
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        print(dataset)
        print(gaussians._opacity.shape)
        # exit(0)
        # scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        print(gaussians._texture.shape)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # t = [[0,1,2],[2,5,6],[1,1,1],[0.5,0.5,0.5]]
        # # t = [[0,2,1,0.5],[1,5,1,0.5],[2,6,1,0.5]]
        # # Repeat the array to form a matrix of shape (100, 12)
        # import numpy as np
        # tensor_np = np.tile(t, (gaussians._opacity.shape[0], 1, 1))
        # print("np", tensor_np.shape, gaussians._features_rest.shape)
        # # Convert the numpy array to a PyTorch tensor
        # gaussians._texture = torch.tensor(tensor_np, device="cuda")
        print("Saving")
        # Save the target point cloud
        point_cloud_path = os.path.join(args.model_path, "point_cloud/iteration_{}".format(50000))
        region_bounds=(-99, 2, -99,99,-99,99)
        gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))



        # if not skip_train:
        #      render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        # if not skip_test:
        #      render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)




from plyfile import PlyData
import numpy as np

def load_and_downsample_point_cloud_ply(filename, voxel_size):
    """
    Load a point cloud from a PLY file, downsample it, and handle features using PlyData.read().

    Args:
    filename (str): Path to the input point cloud file.
    voxel_size (float): The size of the voxel grid to use for downsampling.

    Returns:
    numpy.ndarray: Downsampled points and optionally any additional features.
    """
    # Load the PLY file
    plydata = PlyData.read(filename)
    points = np.vstack([plydata['vertex'][dimension] for dimension in ['x', 'y', 'z']]).T
    
    # Assume additional features might be stored under custom names, handle dynamically
    features = []
    feature_names = ['nx', 'ny', 'nz', 'red', 'green', 'blue', 'alpha']  # Example features
    for name in feature_names:
        if name in plydata['vertex']._property_lookup:
            features.append(plydata['vertex'][name])
    features = np.vstack(features).T if features else None

    # Perform voxel downsampling
    downsampled_points, downsampled_features = custom_voxel_downsample(points, features, voxel_size)

    return downsampled_points, downsampled_features

def custom_voxel_downsample(points, features, voxel_size):
    # Compute voxel indices
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)
    # Convert voxel indices to a unique scalar for each voxel
    voxel_keys = voxel_indices[:, 0] + voxel_indices[:, 1] * 1e3 + voxel_indices[:, 2] * 1e6
    # Find unique voxels and their first occurrence to determine the representative point
    unique_keys, inverse_indices = np.unique(voxel_keys, return_inverse=True)
    # Aggregate points and features in each voxel
    downsampled_points = np.array([points[inverse_indices == i].mean(axis=0) for i in range(len(unique_keys))])
    downsampled_features = np.array([features[inverse_indices == i].mean(axis=0) for i in range(len(unique_keys))]) if features is not None else None
    return downsampled_points, downsampled_features








if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    # print("Rendering " + args.model_path)

    # # Initialize system state (RNG)
    # safe_state(args.quiet)

   

    # render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)


    # Example Usage
    filename = 'point_cloud.ply'
    voxel_size = 0.05  # Define an appropriate voxel size
    downsampled_points, downsampled_features = load_and_downsample_point_cloud_ply(filename, voxel_size)
    print("Downsampled points:", downsampled_points.shape)
    if downsampled_features is not None:
        print("Downsampled features:", downsampled_features.shape)