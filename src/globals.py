# Created on Thu Apr 18 2024 by Florian Pfleiderer
# Copyright (c) 2024 TU Wien
"""
Stores all global variables for the project.

Dataset_Folder is relative to the project root.
"""
import numpy as np

TEST_FOLDER = "data/annotation/"
DATASET_FOLDER = "data/ObChange/"
IMAGE_FOLDER = "data/GH30_Office/"
ROOM = "Office/"
SCENE = "scene2/"
PLANE = "planes/2/"
PCD_PATH = "merged_plane_clouds_ds002.pcd"
ANNO_PATH = "merged_plane_clouds_ds002_GT.anno"
CAMERA_INFO_JSON_PATH = "camera_info.json"
VIEWPOINT_INFO_JSON_PATH = "viewpoint_info.json"
VIEWPOINT_INFO_YAML_PATH = "transformations.yaml"
FOV_X = 60
FOV_Y = 46
GT_COLOR = np.array([0.1, 0.9, 0.1])
MODEL_IMAGE_SIZE = 224
BBOX_AREA = 200
CONFIDENCE_THRESHOLD = 0.20
MAX_PREDICTIONS = 5
