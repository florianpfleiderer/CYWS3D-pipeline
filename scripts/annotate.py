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
        CAMERA_INFO_JSON_PATH, GT_COLOR, MODEL_IMAGE_SIZE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# intrinsic matrix
intrinsics = Intrinsic()
intrinsics.from_json("./" + DATASET_FOLDER + CAMERA_INFO_JSON_PATH)

# create mesh for showing the origin
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
tfs_temp = "BigRoom"
# traverse dataset by scene
for folder in sorted(os.listdir(DATASET_FOLDER)):
    if "Store" in folder or "readme" in folder or "camera_info" in folder:
        continue
    all_target_bboxes = []
    for root, dirnames, files in sorted(os.walk(os.path.join(DATASET_FOLDER, folder))):
        if not 'planes/0' in root:
            continue
        logger.info("Processing folder: %s", root)
        target_bboxes = []
        tfs = root.split("/")
        img_path = "./data/GH30_"+tfs[2]+"/"+tfs[3]+"/"
        try:
            transformations = utils.load_transformations(f"{img_path}{tfs[3]}_transformations.yaml")
        except FileNotFoundError:
            logger.error("No transformation file found in %s", img_path)
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
            img_resized_bboxes = []
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

                resized_u, resized_v = utils.resize_coordinates(gt_u, gt_v, \
                    (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE))

                img_resized_bboxes.append(utils.extract_bboxes(resized_u, resized_v))
                img_bboxes.append(utils.extract_bboxes(gt_u, gt_v))

            all_target_bboxes.append(dict(
                # image_name=img_path+value['file_name'],
                boxes=torch.as_tensor(img_resized_bboxes, dtype=torch.float32),
                labels=torch.zeros((len(img_resized_bboxes),), dtype=torch.int32)
            ))

            final_image = utils.draw_2d_bboxes_on_img(final_image, img_bboxes)
            original_image = utils.draw_2d_bboxes_on_img(
                    f"{img_path}{value['file_name']}", img_bboxes)
            if not path.exists(f"{img_path}ground_truth"):
                os.makedirs(f"{img_path}ground_truth")
            plt.imsave(f"{img_path}ground_truth/{value['file_name']}", original_image)
            plt.imsave(f"{img_path}ground_truth/image_"+key.__str__()+".png", final_image)
            logger.info("image_%s saved as image_%s.png", key, key)

    if len(all_target_bboxes) == 0:
        continue
    torch.save(all_target_bboxes, f"./data/GH30_{folder}/all_target_bboxes.pt")
    logger.info("all_target_bboxes.pt saved in %s", f"./data/GH30_{folder}/all_target_bboxes.pt")
