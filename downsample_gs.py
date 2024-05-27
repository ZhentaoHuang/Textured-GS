import open3d as o3d

# Load your point cloud
point_cloud = o3d.io.read_point_cloud("point_cloud.ply")

# Voxel downsampling
voxel_size = 0.04  # specify voxel size
downsampled_point_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size)

# Save the downsampled point cloud
o3d.io.write_point_cloud("downsampled_point_cloud.ply", downsampled_point_cloud)

# Optional: Confirm by loading and visualizing the saved file
saved_point_cloud = o3d.io.read_point_cloud("downsampled_point_cloud.ply")
o3d.visualization.draw_geometries([saved_point_cloud])
