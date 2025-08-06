from projects.Uni3DETR.uni3detr.voxelization import VoxelizationByGridShape
from mmdet3d.models.voxel_encoders import HardSimpleVFE
import torch
import numpy as np

point_cloud_range = [-54, -54, -5.0, 54, 54, 3.0]
pts_voxel_size = [0.2, 0.2, 0.4]

points = np.random.rand(10000, 5).astype(np.float32) * 100 - 50

voxel_layer = VoxelizationByGridShape(
    max_num_points=10,
    point_cloud_range=point_cloud_range,
    voxel_size=pts_voxel_size,
    max_voxels=(30000, 40000),
    deterministic=False
)

voxels, coords, num_points = voxel_layer(points)

encoder = HardSimpleVFE(num_features=5)
voxel_features = encoder(torch.tensor(voxels), torch.tensor(num_points))
print(voxel_features.shape)  # [num_voxels, C]
