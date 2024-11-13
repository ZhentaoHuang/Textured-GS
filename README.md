# Textured-GS: Gaussian Splatting with Spatially Defined Color and Opacity
Zhentao Huang, Minglun Gong

 [Full Paper](https://arxiv.org/abs/2407.09733)
 [Trained Model](https://drive.google.com/file/d/1at3lJv4-R6PjpAKAk-pjyXdO9_RrxPUp/view?usp=sharing)

# Setting up
Please refer to [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) for setting up. For the Textured-GS submodule:

```shell
pip install submodules/diff-gaussian-rasterization_texture/
```

# Running the code
```shell
python full_eval.py --output_path <mini-spaltting-results (Trained Model)> --mipnerf360 <mipnerf360 folder> --tanksandtemples <tanks and temples folder> --deepblending <deep blending folder>
```
We select [Mini-Splatting](https://github.com/fatPeter/mini-splatting) as the input of our algorithm, which is stored in iteration_66 in point_cloud folder. The default output of Textured-GS is stored in iteration_14000 folder.


## Acknowledgement.
This project is built upon [3DGS](https://github.com/graphdeco-inria/gaussian-splatting).
