import numpy as np
import open3d as o3d

# Load the point cloud
pcd = o3d.io.read_point_cloud("points3D.ply")

# Function to filter the k nearest points to a specified 3D coordinate and preserve additional data
def filter_nearest_points_to_coord(cloud, target_coord, k):
    """
    Filters the k nearest points to a specified 3D coordinate and preserves color and normal data.

    Args:
    cloud (o3d.geometry.PointCloud): The input point cloud.
    target_coord (tuple): The target 3D coordinate (x, y, z).
    k (int): Number of nearest points to find.

    Returns:
    o3d.geometry.PointCloud: A new point cloud containing only the k nearest points with preserved attributes.
    """
    # Convert point cloud to numpy array for points, colors, and normals
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors) if cloud.has_colors() else None
    normals = np.asarray(cloud.normals) if cloud.has_normals() else None
    
    # Calculate distances to the target coordinate
    distances = np.linalg.norm(points - np.array(target_coord), axis=1)
    
    # Get indices of the k smallest distances
    k_nearest_indices = np.argsort(distances)[:k]
    
    # Filter points based on these indices
    k_nearest_points = points[k_nearest_indices]
    
    # Create a new point cloud object with the k nearest points
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(k_nearest_points)
    
    # Also set colors and normals if available
    if colors is not None:
        filtered_pcd.colors = o3d.utility.Vector3dVector(colors[k_nearest_indices])
    if normals is not None:
        filtered_pcd.normals = o3d.utility.Vector3dVector(normals[k_nearest_indices])

    return filtered_pcd

# Parameters
target_coord = (3.5, 0.5, 3.5)  # Target 3D coordinate
k = 1000                          # Number of nearest points to find

# Filter the point cloud
filtered_pcd = filter_nearest_points_to_coord(pcd, target_coord, k)

# Save the filtered point cloud
o3d.io.write_point_cloud("filtered_point_cloud.ply", filtered_pcd)

# Optionally visualize the filtered point cloud
o3d.visualization.draw_geometries([filtered_pcd])
