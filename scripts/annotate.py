#! usr/bin/env python3
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
import copy
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from src.annotation_pipeline.projection \
    import Intrinsic, Extrinsic, project_to_2d, frustum_culling
from src.annotation_pipeline import utils
from src.globals \
    import DATASET_FOLDER, ROOM, SCENE, PLANE, PCD_PATH, ANNO_PATH, CAMERA_INFO_JSON_PATH, GT_COLOR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)

# TODO: traverse dataset
# for root, dirnames, files in os.walk(DATASET_FOLDER):
    # loop over dataset to extract ground truth for every scene containing a
    # transformations.yaml file

transformations = utils.load_transformations("./" + DATASET_FOLDER + ROOM + SCENE \
    + 'transformations.yaml')
# load pcd file
base_pcd = o3d.io.read_point_cloud("./" + DATASET_FOLDER + ROOM + SCENE + PLANE + PCD_PATH)
# ground truth
_, base_anno_dict = utils.annotate_pcd(base_pcd, "./" + DATASET_FOLDER + ROOM + SCENE + PLANE \
    + ANNO_PATH, GT_COLOR)
# intrinsic matrix
intrinsics = Intrinsic()
intrinsics.from_json("./" + DATASET_FOLDER + CAMERA_INFO_JSON_PATH)

final_image = np.zeros((intrinsics.height, intrinsics.width, 3), dtype=np.uint8)

# create mesh for showing the origin
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
if logger.level == logging.DEBUG:
    bboxes = utils.extract_3d_bboxes(base_pcd, base_anno_dict, result=False)
    o3d.visualization.draw_geometries([base_pcd, bboxes[0], bboxes[1], bboxes[2]])

for key, value in transformations.items():
    anno_dict = copy.deepcopy(base_anno_dict)
    # extrinsic matrix
    extrinsics = Extrinsic()
    extrinsics.from_dict(value)

    for anno_key, anno_value in anno_dict.items():
        logger.info("Annotating object: %s", anno_key)
        pcd = copy.deepcopy(base_pcd)

        # pcd.transform(M)
        pcd.transform(extrinsics.homogenous_matrix())
        points_pos = np.asarray(pcd.points)
        points_color = np.asarray(pcd.colors)
        if logger.level == logging.DEBUG:
            o3d.visualization.draw_geometries([pcd, mesh_frame])
        gt_pcd = pcd.select_by_index(anno_value)
        if logger.level == logging.DEBUG:
            o3d.visualization.draw_geometries([gt_pcd])
        gt_points_pos = np.asarray(gt_pcd.points)
        gt_points_color = np.asarray(gt_pcd.colors)

        # frustum culling
        try:
            pcd = pcd.select_by_index(frustum_culling(points_pos, 60))
            gt_pcd = gt_pcd.select_by_index(frustum_culling(gt_points_pos, 60))
        except ValueError:
            logger.warning("Object %s not found in frustum, continuing with next object", anno_key)
            continue

        points_pos = np.asarray(pcd.points)
        points_color = np.asarray(pcd.colors)
        gt_points_pos = np.asarray(gt_pcd.points)
        gt_points_color = np.asarray(gt_pcd.colors)

        # project and draw bboxes
        u_coords, v_coords = project_to_2d(points_pos, \
                                            intrinsics.homogenous_matrix(), \
                                            intrinsics.width, \
                                            intrinsics.height)

        gt_u, gt_v = project_to_2d(gt_points_pos, intrinsics.homogenous_matrix(), \
            intrinsics.width, intrinsics.height)

        final_image = utils.draw_image((u_coords, v_coords), points_color, (gt_u, gt_v), intrinsics)
        final_image = utils.draw_2d_bboxes_on_img(final_image, gt_u, gt_v)

        # image = utils.draw_2d_bboxes_on_img(\
        #   "./data/annotation/office/scene4/img03_s4.png", gt_u, gt_v)
    # except AssertionError:
    #     logger.error("No object found in frustum")
    #     continue

    plt.imsave(f"image_{key}.png", final_image)
    logger.info("image_%f saved as image_%f.png", key, key)
