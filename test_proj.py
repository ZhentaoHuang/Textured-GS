import numpy as np

def pixel_to_ray(pixel, screen_dimensions, proj_matrix):
    width, height = screen_dimensions
    x, y = pixel

    # Convert pixel coordinates to NDC
    ndc_x = (2.0 * (x + 0.5) / width) - 1.0
    ndc_y = 1.0 - (2.0 * (y + 0.5) / height)  # Flip y for graphics coordinates
    ndc_z = -0.01  # Assuming the near plane value for z
    ndc_w = 1.0  # Homogeneous coordinate

    ndc = np.array([ndc_x, -ndc_y, ndc_z, ndc_w])

    # Transform to clip space using the inverse projection matrix
    inv_proj_matrix = np.linalg.inv(proj_matrix)
    np.set_printoptions(suppress=True, precision=6)
    print(inv_proj_matrix)
    world_coords = inv_proj_matrix.dot(ndc)

    # Normalize the resulting vector (ignoring w)
    norm = np.linalg.norm(world_coords[:3])
    
    ray = world_coords[:3] / world_coords[3]

    return ndc[:3], world_coords, ray, norm

def test_pixel_patch(top_left, bottom_right, screen_dimensions, proj_matrix):
    for y in range(top_left[1], bottom_right[1] + 1):
        for x in range(top_left[0], bottom_right[0] + 1):
            ndc, world_coords, ray, norm = pixel_to_ray((x, y), screen_dimensions, proj_matrix)
            print(f"Pixel: ({x}, {y}), NDC: {ndc}, World: {world_coords}, Ray: {ray}, norm: {norm}")


# Example projection matrix
proj_matrix = np.array([
[0.937958, -0.037863,  0.729438, -1.397136],
[-0.022941, 2.114965, 0.139281, -1.088215],
[-0.614377, -0.058490, 0.786969, 4.056883],
[-0.614315, -0.058484, 0.786890, 4.066477]
])

view_matrix = np.array([
[0.788986, -0.031849, 0.613585, -1.175235],
[-0.010823, 0.997780, 0.065709, -0.513388],
[ -0.614315, -0.058484, 0.786890, 4.066477],
[0.000000, 0.000000, 0.000000, 1.000000]
])


rot = ()
mean = (3.4198, 0.7126, -2.4450)

# Screen dimensions
screen_dimensions = (979, 546)

# Define the pixel patch coordinates
top_left = (346, 264)
bottom_right = (347, 265)

# Test the patch
test_pixel_patch(top_left, bottom_right, screen_dimensions, proj_matrix)


