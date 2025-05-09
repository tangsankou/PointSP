import os
import numpy as np
import open3d as o3d

# points = np.random.rand(10000, 3)
point_cloud = o3d.io.read_point_cloud("/home/user_tp/workspace/data/ModelNet40/airplane/test/airplane_0627.ply")
o3d.visualization.draw_geometries([point_cloud])
