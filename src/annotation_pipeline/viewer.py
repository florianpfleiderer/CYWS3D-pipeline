# Created on Thu Apr 18 2024 by Florian Pfleiderer
# Copyright (c) 2024 TU Wien
"""
used to View pointclouds and annotations from ObChange dataset
"""
import open3d as o3d
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
try:
    from src.globals import DATASET_FOLDER, ROOM, SCENE, PLANE, PCD_PATH 
    PREFIX = "../../"
except ImportError:
    from ..globals import DATASET_FOLDER, ROOM, SCENE, PLANE, PCD_PATH
    PREFIX = "../../"

# load pcd file
pcd = o3d.io.read_point_cloud(PREFIX + DATASET_FOLDER + ROOM + SCENE + PLANE + PCD_PATH)
# create mesh for showing the origin
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])

o3d.visualization.draw_geometries([pcd, mesh_frame])
