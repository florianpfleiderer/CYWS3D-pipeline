# Created on Thu Apr 18 2024 by Florian Pfleiderer
# Copyright (c) 2024 TU Wien
"""
Stores all global variables for the project.

Dataset_Folder is relative to the project root.
"""
import numpy as np

# annotate.py
TEST_FOLDER = "data/annotation/"
DATASET_FOLDER = "data/ObChange/"
IMAGE_FOLDER = "data/GH30_Office/"
ROOM = "Office/"
SCENE = "scene2/"
PLANE = "planes/0/"
PCD_PATH = "merged_plane_clouds_ds002.pcd"
ANNO_PATH = "merged_plane_clouds_ds002_GT.anno"
CAMERA_INFO_JSON_PATH = "camera_info.json"
VIEWPOINT_INFO_JSON_PATH = "viewpoint_info.json"
VIEWPOINT_INFO_YAML_PATH = "transformations.yaml"
GT_COLOR = np.array([0.1, 0.1, 0.9])
