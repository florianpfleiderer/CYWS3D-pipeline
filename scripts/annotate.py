#! usr/bin/env python3
# Created on Thu Apr 18 2024 by Florian Pfleiderer
# Copyright (c) 2024 TU Wien
"""
Annotate images with ground truth bounding boxes.

This script loads the ground truth annotations from the merged_plane_clouds_ds002_GT.anno file
and projects the 3D bounding boxes to 2D image space. The 2D bounding boxes are then resized
to the model image size and saved in the ground_truth folder of the image folder.
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
        CAMERA_INFO_JSON_PATH, GT_COLOR, MODEL_IMAGE_SIZE, FOV_X, FOV_Y, BBOX_AREA
from src.modules.geometry import remove_bboxes_with_area_less_than

logging.basicConfig()
logger = logging.getLogger(__name__)

def main(
    log_level: str = "INFO", 
    room: str = "ALL",
    bbox_area: int=BBOX_AREA):
    """
    main function for annotation pipeline

    Args:
        log_level (str): logging level
        room (str): room to process, default is all rooms
    """
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.warning("logger set to %s", logger.level)

    # intrinsic matrix
    intrinsics = Intrinsic()
    intrinsics.from_json("./" + DATASET_FOLDER + CAMERA_INFO_JSON_PATH)

    # create mesh for showing the origin
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    tfs_temp = "BigRoom"
    scene_buffer = None
    # traverse dataset by scene
    for folder in sorted(os.listdir(DATASET_FOLDER)):
        if "Store" in folder or "readme" in folder or "camera_info" in folder:
            continue
        all_target_bboxes = []
        for root, dirnames, files in sorted(os.walk(os.path.join(DATASET_FOLDER, folder))):
            if not 'planes/' in root:
                continue
            if not 'merged_plane_clouds_ds002_GT.anno' in files:
                continue
            if not room == "ALL" and not room in root:
                continue

            logger.info("Processing folder: %s", root)
            logger.debug("Files: %s", files)
            target_bboxes = []
            tfs = root.split("/")
            img_path = "data/GH30_"+tfs[2]+"/"+tfs[3]+"/"
            logger.debug("Image path: %s", img_path)
            try:
                transformations = utils.load_transformations(f"{img_path}{tfs[3]}_transformations.yaml")
            except FileNotFoundError:
                logger.warning("No transformation file found in %s", img_path)
                continue
            # load pcd file
            logger.debug("loading pointcloud from %s", "./"+root+"/"+PCD_PATH)
            base_pcd = o3d.io.read_point_cloud(f"./{root}/{PCD_PATH}")
            
            # ground truth
            logger.debug("loading ground truth from %s", "./"+root+"/"+ANNO_PATH)
            if scene_buffer != tfs[3]:
                scene_buffer = tfs[3]
                logger.info("\n#####################################################\n")
                logger.info("New scene: %s", scene_buffer)
                scene_annotation_buffer = {}
            
            _, base_anno_dict = utils.annotate_pcd(base_pcd, f"./{root}/{ANNO_PATH}", GT_COLOR)

            if logger.level == logging.DEBUG:
                bboxes = utils.extract_3d_bboxes(base_pcd, base_anno_dict, result=False)
                o3d.visualization.draw_geometries([base_pcd, *bboxes])

            # iterate over images in each scene
            for key, value in transformations.items():
                logger.info("Processing image: %s", key)
                dict_entry_missing = True
                skip_loop = True
                for image_bboxes in all_target_bboxes:
                    if image_bboxes['image'] == img_path+value['file_name']:
                        dict_entry_missing = False
                        break
                if dict_entry_missing:
                    logger.info("Creating new dict entry for image %s", key)
                    all_target_bboxes.append(dict(
                        image=img_path+value['file_name'],
                        boxes=torch.as_tensor([], dtype=torch.float32),
                        labels=torch.zeros((0,), dtype=torch.int32)
                    ))
                extrinsics = Extrinsic()
                extrinsics.from_dict(value)
                pcd = copy.deepcopy(base_pcd)
                pcd.transform(extrinsics.homogenous_matrix())
                base_gt_pcd = copy.deepcopy(pcd)
                points_pos = np.asarray(pcd.points)
                points_color = np.asarray(pcd.colors)
                try:
                    pcd = pcd.select_by_index(frustum_culling(points_pos, FOV_X, FOV_Y))
                except ValueError:
                    logger.warning("No points in frustum for plane %s, skipping", tfs[5])
                    continue
                if logger.level == logging.DEBUG:
                    o3d.visualization.draw_geometries([pcd, mesh_frame])
                points_pos = np.asarray(pcd.points)
                points_color = np.asarray(pcd.colors)

                u_coords, v_coords = project_to_2d(
                    points_pos, intrinsics.homogenous_matrix(), intrinsics.distortion_coeffs(), \
                        intrinsics.width, intrinsics.height)
                final_image = utils.draw_image((u_coords, v_coords), points_color, intrinsics)

                # iterate over ground truth objects and extract bboxes for visible objects
                # img_bboxes = []
                img_resized_bboxes = []
                anno_dict = copy.deepcopy(base_anno_dict)
                for anno_key, anno_value in base_anno_dict.items():
                    logger.info("Annotating object: %s", anno_key)
                    if key in scene_annotation_buffer:
                        if anno_key in scene_annotation_buffer[key]:
                            logger.info("Object %s already annotated on image %s", anno_key, key)
                            continue
                    gt_pcd = base_gt_pcd.select_by_index(anno_value)
                    if logger.level == logging.DEBUG:
                        o3d.visualization.draw_geometries([gt_pcd])
                    gt_points_pos = np.asarray(gt_pcd.points)
                    gt_points_color = np.asarray(gt_pcd.colors)
                    try:
                        gt_pcd = gt_pcd.select_by_index(frustum_culling(gt_points_pos, FOV_X, FOV_Y))
                    except ValueError:
                        logger.warning("Object %s not found in frustum, continuing with next object",\
                            anno_key)
                        continue

                    gt_points_pos = np.asarray(gt_pcd.points)
                    gt_points_color = np.asarray(gt_pcd.colors)

                    gt_u, gt_v = project_to_2d(gt_points_pos, intrinsics.homogenous_matrix(), \
                        intrinsics.distortion_coeffs(), intrinsics.width, intrinsics.height)

                    resized_u, resized_v = utils.resize_coordinates(gt_u, gt_v, \
                        (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE))

                    img_resized_bboxes.append(utils.extract_bboxes(resized_u, resized_v))
                    # img_bboxes.append(utils.extract_bboxes(gt_u, gt_v))

                    if not key in scene_annotation_buffer:
                        scene_annotation_buffer[key] = []
                    scene_annotation_buffer[key].append(anno_key)
                    skip_loop = False
                    logger.info("Object %s annotated on image %s", anno_key, key)

                if skip_loop:
                    logger.info("Skipping annotation for image %s", value['file_name'])
                    continue
                img_resized_bboxes = remove_bboxes_with_area_less_than(
                    np.asarray(img_resized_bboxes), bbox_area)

                for all_img_bboxes in all_target_bboxes:
                    if all_img_bboxes['image'] == img_path+value['file_name']:
                        all_img_bboxes['boxes'] = torch.cat((all_img_bboxes['boxes'], \
                            torch.as_tensor(img_resized_bboxes, dtype=torch.float32)))
                        all_img_bboxes['labels'] = torch.cat((all_img_bboxes['labels'], \
                            torch.zeros((len(img_resized_bboxes),), dtype=torch.int32)))

                # img_bboxes = np.asarray(img_bboxes)
                new_img_bboxes = np.zeros_like(img_resized_bboxes).astype(np.int32)
                new_img_bboxes[:, 2] = img_resized_bboxes[:, 2] / MODEL_IMAGE_SIZE * intrinsics.width
                new_img_bboxes[:, 0] = img_resized_bboxes[:, 0] / MODEL_IMAGE_SIZE * intrinsics.width
                new_img_bboxes[:, 3] = img_resized_bboxes[:, 3] / MODEL_IMAGE_SIZE * intrinsics.height
                new_img_bboxes[:, 1] = img_resized_bboxes[:, 1] / MODEL_IMAGE_SIZE * intrinsics.height

                final_image = utils.draw_2d_bboxes_on_img(final_image, new_img_bboxes)
                if not path.exists(f"{img_path}ground_truth"):
                    os.makedirs(f"{img_path}ground_truth")
                if not path.exists(f"{img_path}ground_truth/{value['file_name']}"):
                    original_image = utils.draw_2d_bboxes_on_img(
                        f"{img_path}{value['file_name']}", new_img_bboxes)
                else:
                    original_image = utils.draw_2d_bboxes_on_img(
                        f"{img_path}ground_truth/{value['file_name']}", new_img_bboxes)
                plt.imsave(f"{img_path}ground_truth/{value['file_name']}", original_image)

                plt.imsave(
                    f"{img_path}ground_truth/image_{tfs[4]}{tfs[5]}_{key}.png", \
                        final_image)
                logger.info("image_%s saved as image_%s.png", \
                    key, f"{tfs[4]}{tfs[5]}_{key.__str__()}")

        if len(all_target_bboxes) == 0:
            continue
        torch.save(all_target_bboxes, f"./data/GH30_{folder}/all_target_bboxes.pt")
        logger.info("all_target_bboxes.pt saved in %s", \
            f"./data/GH30_{folder}/all_target_bboxes.pt")


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
