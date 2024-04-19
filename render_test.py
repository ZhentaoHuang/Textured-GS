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
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        print(gaussians._opacity.shape)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


        ############
        # gaussians._scaling = gaussians._scaling - 0.693
        scales = gaussians.get_scaling.cpu()
        # scales = torch.randn(5,3)
        sorted_scales, indices = torch.sort(scales, dim=1, descending=True)
        # sorted_scales = scales
        x_scales = sorted_scales[:, 0]
        y_scales = sorted_scales[:, 1]
        z_scales = sorted_scales[:, 2]
        w_scales = x_scales / z_scales
        h_scales = x_scales / y_scales
        
        # Compute statistics
        x_mean, y_mean, z_mean, w_mean = x_scales.mean(), y_scales.mean(), z_scales.mean(), w_scales.mean()
        x_std, y_std, z_std, w_std = x_scales.std(), y_scales.std(), z_scales.std(), w_scales.std()
        x_median, y_median, z_median, w_median = x_scales.median(), y_scales.median(), z_scales.median(), w_scales.median()

        # Compute min and max for each axis
        x_min, x_max = x_scales.min(), x_scales.max()
        y_min, y_max = y_scales.min(), y_scales.max()
        z_min, z_max = z_scales.min(), z_scales.max()
        w_min, w_max = w_scales.min(), w_scales.max()

        # Print min and max values
        print(f"X Scale: Min = {x_min}, Max = {x_max}")
        print(f"Y Scale: Min = {y_min}, Max = {y_max}")
        print(f"Z Scale: Min = {z_min}, Max = {z_max}")
        print(f"W Scale: Min = {w_min}, Max = {w_max}")

        print(f"Means: X={x_mean}, Y={y_mean}, Z={z_mean}, W={w_mean}")
        print(f"Standard Deviations: X={x_std}, Y={y_std}, Z={z_std}, W={w_std}")
        print(f"Medians: X={x_median}, Y={y_median}, Z={z_median}, W={w_median}")




        thresholds = [2, 10, 50, 100, 1000]

        # Calculate the total number of values in w_scales
        total_values = w_scales.numel()

        percentages_w, percentages_h = [],[]

        for threshold in thresholds:
            # Count the number of values above the current threshold
            above_threshold = torch.sum(w_scales > threshold)
            above_threshold_h = torch.sum(h_scales > threshold)

            # Calculate the percentage of values above the current threshold
            percentage_above_threshold = (above_threshold.float() / total_values) * 100
            percentages_w.append(percentage_above_threshold)
            print(f"Percentage of w_scales values above {threshold}: {percentage_above_threshold:.2f}%")

            percentage_above_threshold = (above_threshold_h.float() / total_values) * 100
            percentages_h.append(percentage_above_threshold)
            print(f"Percentage of h_scales values above {threshold}: {percentage_above_threshold:.2f}%")





        import matplotlib.pyplot as plt

        # Define a helper function to add labels on top of each bar
        def add_labels(ax, bars, percentages):
            for bar, percentage in zip(bars, percentages):
                height = bar.get_height()
                ax.annotate(f'{percentage:.2f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')



       # Creating a figure with 2 subplots (1 row, 2 columns)
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Convert thresholds to string labels for plotting in the first subplot
        threshold_labels = [f">{threshold}" for threshold in thresholds]

        
        # Subplot 1: Bar chart of percentages above thresholds for h_scales
        bars_h = axs[0].bar(threshold_labels, percentages_h, color='skyblue')
        axs[0].set_xlabel('Thresholds')
        axs[0].set_ylabel('Percentage Above Threshold')
        axs[0].set_title('x/y: % Above Thresholds')
        add_labels(axs[0], bars_h, percentages_h)

        # Subplot 2: Bar chart of percentages above thresholds for w_scales
        bars_w = axs[1].bar(threshold_labels, percentages_w, color='orange')
        axs[1].set_xlabel('Thresholds')
        axs[1].set_ylabel('Percentage Above Threshold')
        axs[1].set_title('x/z: % Above Thresholds')
        add_labels(axs[1], bars_w, percentages_w)

        # Adjust layout for better spacing
        plt.tight_layout()
        plt.savefig('h_w_scales_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
  
        ## Save the target point cloud
        # point_cloud_path = os.path.join(args.model_path, "point_cloud/iteration_{}".format(50000))
        # gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))



        # if not skip_train:
        #      render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        # if not skip_test:
        #      render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

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
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

   

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)