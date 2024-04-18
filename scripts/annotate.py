# Created on Thu Apr 18 2024 by Florian Pfleiderer
# Copyright (c) 2024 TU Wien
"""
module for office dataset 

    - load point cloud
    - load annotation
    - load camera info
    - load viewpoint info
    - project to 2d
    - draw 2d bboxes
    - show image
"""
import logging
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from src.annotation_pipeline.projection import Intrinsic, Extrinsic, project_to_2d, frustum_culling
from src.annotation_pipeline import utils
from src.globals import DATASET_FOLDER, ROOM, SCENE, PLANE, PCD_PATH, ANNO_PATH, CAMERA_INFO_JSON_PATH, VIEWPOINT_INFO_JSON_PATH, VIEWPOINT_INFO_YAML_PATH, GT_COLOR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# TODO: traverse dataset
# for root, dirnames, files in os.walk(DATASET_FOLDER):
    # loop over dataset to extract ground truth for every scene containing a 
    # transformations.yaml file

# load pcd file
pcd = o3d.io.read_point_cloud("./" + DATASET_FOLDER + ROOM + SCENE + PLANE + PCD_PATH)

# ground truth
_, anno_dict = utils.annotate_pcd(pcd, "./" + DATASET_FOLDER + ROOM + SCENE + PLANE + ANNO_PATH, GT_COLOR)
# bboxes = utils.extract_3d_bboxes(pcd, anno_dict, result=False)
# o3d.visualization.draw_geometries([pcd, bboxes[0], bboxes[1], bboxes[2]])
# if True:
#     exit()

# intrinsic matrix
intrinsics = Intrinsic()
intrinsics.from_json("./" + DATASET_FOLDER + CAMERA_INFO_JSON_PATH)

# extrinsic matrix
extrinsics = Extrinsic()
extrinsics.from_yaml("./" + DATASET_FOLDER + ROOM + SCENE + VIEWPOINT_INFO_YAML_PATH)

# create mesh for showing the origin
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])

# pcd.transform(M)
pcd.transform(extrinsics.homogenous_matrix())
points_pos = np.asarray(pcd.points)
points_color = np.asarray(pcd.colors)
o3d.visualization.draw_geometries([pcd, mesh_frame])
if True:
    exit()


# indices change after "select by index" (new pcd is created)
# gt_pcd = pcd.select_by_index(anno_dict['077_rubiks_cube'])
gt_pcd = pcd.select_by_index(anno_dict['025_mug'])
gt_points_pos = np.asarray(gt_pcd.points)
gt_points_color = np.asarray(gt_pcd.colors)

# frustum culling
pcd = pcd.select_by_index(frustum_culling(points_pos, 60))
gt_pcd = gt_pcd.select_by_index(frustum_culling(gt_points_pos, 60))

points_pos = np.asarray(pcd.points)
points_color = np.asarray(pcd.colors)
gt_points_pos = np.asarray(gt_pcd.points)
gt_points_color = np.asarray(gt_pcd.colors)

# project and draw bboxes
u_coords, v_coords = project_to_2d(points_pos, \
                                    intrinsics.homogenous_matrix(), \
                                    intrinsics.width, \
                                    intrinsics.height)

gt_u, gt_v = project_to_2d(gt_points_pos, intrinsics.homogenous_matrix(), intrinsics.width, intrinsics.height)

image = utils.draw_2d_bboxes((u_coords, v_coords), points_color, (gt_u, gt_v), intrinsics)

# image = utils.draw_2d_bboxes_on_img("./data/annotation/office/scene4/img03_s4.png", gt_u, gt_v)

plt.imsave("image.png", image)
logger.info("image saved as image.png")

