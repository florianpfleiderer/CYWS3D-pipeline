import open3d as o3d
import open3d.core as o3c
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json


folder = "../data/annotation/office/"
scene = "scene4/"
pcd_path = "merged_plane_clouds_ds002.pcd"
anno_path = "merged_plane_clouds_ds002_GT.anno"
camera_info_json_path = "camera_info.json" 
viewpoint_info_json_path = "viewpoint_info.json"
gt_colour = np.array([0.1, 0.90, 0.1])

# load pcd file
pcd = o3d.io.read_point_cloud("./" + folder + scene + pcd_path)
# create mesh for showing the origin
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])

o3d.visualization.draw_geometries([pcd, mesh_frame])