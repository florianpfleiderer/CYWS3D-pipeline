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
import os
from os import path
import logging
import copy
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from src.annotation_pipeline.projection \
    import Intrinsic, Extrinsic, project_to_2d, frustum_culling
from src.annotation_pipeline import utils
from src.globals \
    import DATASET_FOLDER, IMAGE_FOLDER, ROOM, SCENE, PLANE, PCD_PATH, ANNO_PATH, \
        CAMERA_INFO_JSON_PATH, GT_COLOR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)

# intrinsic matrix
intrinsics = Intrinsic()
intrinsics.from_json("./" + DATASET_FOLDER + CAMERA_INFO_JSON_PATH)

# create mesh for showing the origin
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
all_target_bboxes = []
tfs_temp = "BigRoom"
# traverse dataset by scene
for root, dirnames, files in sorted(os.walk(DATASET_FOLDER)):
    if not 'planes/0' in root:
        continue
    logger.info("Processing folder: %s", root)
    target_bboxes = []
    tfs = root.split("/")
    img_path = "./data/GH30_"+tfs[2]+"/"+tfs[3]+"/"
    try:
        transformations = utils.load_transformations(f"{img_path}{tfs[3]}_transformations.yaml")
    except FileNotFoundError:
        logger.warning("No transformation file found in %s", img_path)
        tfs_temp = tfs[2]
        logger.info("Updating tfs_temp to %s", tfs_temp)
        continue
    # load pcd file
    base_pcd = o3d.io.read_point_cloud(f"./{root}/{PCD_PATH}")
    # ground truth
    _, base_anno_dict = utils.annotate_pcd(base_pcd, f"./{root}/{ANNO_PATH}", GT_COLOR)

    if logger.level == logging.DEBUG:
        bboxes = utils.extract_3d_bboxes(base_pcd, base_anno_dict, result=False)
        o3d.visualization.draw_geometries([base_pcd, bboxes[0], bboxes[1], bboxes[2]])

    # iterate over images in each scene
    for key, value in transformations.items():
        logger.info("Processing image: %s", key)
        
        extrinsics = Extrinsic()
        extrinsics.from_dict(value)

        pcd = copy.deepcopy(base_pcd)
        pcd.transform(extrinsics.homogenous_matrix())
        base_gt_pcd = copy.deepcopy(pcd)
        points_pos = np.asarray(pcd.points)
        points_color = np.asarray(pcd.colors)
        if logger.level == logging.DEBUG:
            o3d.visualization.draw_geometries([pcd, mesh_frame])

        pcd = pcd.select_by_index(frustum_culling(points_pos, 60))

        points_pos = np.asarray(pcd.points)
        points_color = np.asarray(pcd.colors)

        u_coords, v_coords = project_to_2d(points_pos, \
                                            intrinsics.homogenous_matrix(), \
                                            intrinsics.width, \
                                            intrinsics.height)

        final_image = utils.draw_image((u_coords, v_coords), points_color, intrinsics)

        # iterate over ground truth objects and extract bboxes for visible objects
        img_bboxes = []
        anno_dict = copy.deepcopy(base_anno_dict)
        for anno_key, anno_value in base_anno_dict.items():
            logger.info("Annotating object: %s", anno_key)

            gt_pcd = base_gt_pcd.select_by_index(anno_value)
            if logger.level == logging.DEBUG:
                o3d.visualization.draw_geometries([gt_pcd])
            gt_points_pos = np.asarray(gt_pcd.points)
            gt_points_color = np.asarray(gt_pcd.colors)

            # frustum culling
            try:
                gt_pcd = gt_pcd.select_by_index(frustum_culling(gt_points_pos, 60))
            except ValueError:
                logger.warning("Object %s not found in frustum, continuing with next object",\
                    anno_key)
                continue

            gt_points_pos = np.asarray(gt_pcd.points)
            gt_points_color = np.asarray(gt_pcd.colors)

            gt_u, gt_v = project_to_2d(gt_points_pos, intrinsics.homogenous_matrix(), \
                intrinsics.width, intrinsics.height)

            img_bboxes.append(utils.extract_bboxes(gt_u, gt_v))

            # u_coords, v_coords = utils.squeeze_coordinates(
            #     (u_coords, v_coords), intrinsics, 0.0005)
            # gt_u, gt_v = utils.squeeze_coordinates((gt_u, gt_v), intrinsics, 0.0005)

            # if np.array_equal(final_image, np.zeros(
            #         (intrinsics.height, intrinsics.width, 3), dtype=np.uint8)):
            #     logger.info("Initializing image")
            #     final_image = utils.draw_image((u_coords, v_coords), points_color, intrinsics)
            #     plt.imsave(root+"_image"+key.__str__()+".png", final_image)
            
            # final_image = utils.draw_2d_bboxes_on_img(final_image, gt_u, gt_v)
            # if path.exists(f"{img_path}ground_truth/{value['file_name']}"):
            #     original_image = utils.draw_2d_bboxes_on_img(
            #         f"{img_path}ground_truth/{value['file_name']}", gt_u, gt_v)
            # else:
            #     original_image = utils.draw_2d_bboxes_on_img(
            #         f"{img_path}{value['file_name']}", gt_u, gt_v)
            # plt.imsave(f"{img_path}ground_truth/{value['file_name']}", original_image)

        target_bboxes.append(dict(
            boxes=torch.as_tensor(img_bboxes, dtype=torch.float32),
            labels=torch.ones((len(img_bboxes),), dtype=torch.int64)
        ))

        final_image = utils.draw_2d_bboxes_on_img(final_image, img_bboxes)
        original_image = utils.draw_2d_bboxes_on_img(
                f"{img_path}{value['file_name']}", img_bboxes)
        plt.imsave(f"{img_path}ground_truth/{value['file_name']}", original_image)
        plt.imsave(f"{img_path}ground_truth/image_"+key.__str__()+".png", final_image)
        logger.info("image_%s saved as image_%s.png", key, key)

    torch.save(target_bboxes, f"{img_path}ground_truth/target_bboxes.pt")
    if tfs[2] == tfs_temp:
        all_target_bboxes.append(target_bboxes)
    else:
        # check if all_target_bboxes is empty
        if len(all_target_bboxes) != 0:
            torch.save(all_target_bboxes, f"data/GH30_{tfs_temp}/all_target_bboxes.pt")
            logger.info("all_target_bboxes.pt saved in data/GH30_%s/", tfs_temp)
        # TODO: target bboxes are not saved for office when running the script?
        all_target_bboxes = []
        all_target_bboxes.append(target_bboxes)
    tfs_temp = tfs[2]
    logger.info("target_bboxes.pt saved in %s", img_path)

