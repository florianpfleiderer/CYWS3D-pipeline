# Created on Thu Jul 04 2024 by Florian Pfleiderer
# Copyright (c) 2024 TU Wien
"""
This module contains the functions to create metadata for the inference.
"""
import os
import numpy as np
from src.globals import CAMERA_INFO_JSON_PATH
from src.annotation import utils
from src.annotation.projection import Intrinsic, Extrinsic

def create_metadata(
    ROOM, 
    ROOM_DIR, 
    DEPTH, 
    TRANSFORMATIONS,
    perspective,
    registration_strategy,
    substrings, 
    batch,
    logger
):
    """
    Create metadata for inference.
    """

    scene1_buffer = sorted([f for f in os.listdir(os.path.join(ROOM_DIR, "scene1")) \
        if ".png" in f])

    if ROOM == "LivingArea":
        img_number = 0
        for scene in sorted(os.listdir(ROOM_DIR)):
            if any(substring in scene for substring in substrings):
                logger.debug("Skipping %s", scene)
                continue
            logger.info("Processing Scene %s", scene)
            scene_buffer = []
            logger.debug("substring: %s", substrings[:-1])
            for img in os.listdir(os.path.join(ROOM_DIR, scene)):
                if any(substring in img for substring in substrings[:-1]):
                    continue
                scene_buffer.append(img)
            scene_buffer.sort()
            logger.debug("scene_buffer: %s", scene_buffer)
            if DEPTH:
                if perspective=="3d":
                    logger.info("only depth images with perspective change are added to metadata")
                    batch.append({
                    "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[9]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[8]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[3]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[2]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                    "prediction_number": img_number,
                    "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[8]),
                    "image2": os.path.join(ROOM_DIR, scene, scene_buffer[9]),
                    "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[2]),
                    "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[3]),
                    "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                elif perspective=="2d":
                    logger.info("Adding depth images withour perseption changes to metadata")
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[6]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[6]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[0]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[0]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[7]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[7]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[1]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[1]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[8]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[8]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[2]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[2]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[9]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[9]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[3]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[3]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[10]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[10]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[4]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[4]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[11]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[11]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[5]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[5]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                else:
                    logger.info("adding all depth images")
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[6]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[6]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[0]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[0]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[7]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[7]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[1]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[1]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[9]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[8]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[3]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[2]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[8]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[9]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[2]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[3]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[10]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[10]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[4]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[4]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[11]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[11]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[5]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[5]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
            else:
                logger.info("Only RGB images are added to metadata")
                if perspective=="3d":
                    logger.info("RGB images with perspective change are added to metadata")
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[9]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[8]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[8]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[9]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                elif perspective=="2d":
                    logger.info("Adding RGB images without perseption changes to metadata")
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[6]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[6]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[7]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[7]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[8]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[8]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[9]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[9]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[10]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[10]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[11]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[11]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                else:
                    logger.info("adding all RGB images")
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[6]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[6]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[7]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[7]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[9]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[8]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[8]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[9]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[10]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[10]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[11]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[11]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1

    if ROOM == "Office":
        img_number = 0
        for scene in sorted(os.listdir(ROOM_DIR)):
            if any(substring in scene for substring in substrings):
                continue
            scene_buffer = []
            logger.debug("substring: %s", substrings[:-1])
            for img in os.listdir(os.path.join(ROOM_DIR, scene)):
                if any(substring in img for substring in substrings[:-1]):
                    continue
                scene_buffer.append(img)
            scene_buffer.sort()
            if DEPTH:
                logger.info("Adding depth images to metadata")
                if perspective=="3d":
                    logger.info("only depth images with perspective change are added to metadata")
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[5]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[4]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[1]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[0]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[4]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[5]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[0]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[1]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[7]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[6]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[3]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[2]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[6]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[7]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[2]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[3]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                elif perspective=="2d":
                    logger.info("Adding depth images without perspective changes to metadata")
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[4]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[4]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[0]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[0]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[5]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[5]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[0]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[0]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[6]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[6]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[2]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[2]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[7]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[7]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[3]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[3]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                else:
                    logger.info("adding all depth images")
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[4]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[4]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[0]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[0]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[4]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[5]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[0]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[1]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[6]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[6]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[2]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[2]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[6]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[7]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[2]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[3]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
            else:
                logger.info("Only RGB images are added to metadata")
                if perspective=="3d":
                    logger.info("only RGB images with perspective change are added to metadata")
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[5]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[4]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[4]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[5]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[7]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[6]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[6]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[7]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                elif perspective=="2d":
                    logger.info("Adding RGB images without perspective changes to metadata")
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[4]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[4]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[5]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[5]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[6]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[6]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[7]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[7]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                else:
                    logger.info("adding all RGB images")
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[4]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[4]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[4]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[5]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[6]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[6]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[6]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[7]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1

    if ROOM == "SmallRoom":
        img_number = 0
        for scene in sorted(os.listdir(ROOM_DIR)):
            if any(substring in scene for substring in substrings):
                continue
            scene_buffer = []
            logger.debug("substring: %s", substrings[:-1])
            for img in os.listdir(os.path.join(ROOM_DIR, scene)):
                if any(substring in img for substring in substrings[:-1]):
                    continue
                scene_buffer.append(img)
            scene_buffer.sort()
            logger.debug("scene_buffer: %s", scene_buffer)
            if DEPTH:
                logger.info("Adding depth images to metadata")
                if perspective=="3d":
                    logger.info("perspective changes are added to metadata")
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[10]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[8]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[2]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[0]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[11]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[9]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[3]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[1]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[8]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[10]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[0]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[2]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[9]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[11]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[1]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[3]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[15]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[12]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[7]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[4]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[14]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[13]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[6]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[5]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[13]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[14]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[5]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[6]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[12]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[15]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[4]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[7]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                elif perspective=="2d":
                    logger.info("No perspective changes to metadata")
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[8]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[8]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[0]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[0]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[9]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[9]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[1]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[1]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[10]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[10]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[2]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[2]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[11]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[11]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[3]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[3]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[12]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[12]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[4]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[4]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[13]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[13]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[5]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[5]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[14]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[14]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[6]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[6]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[15]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[15]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[7]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[7]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                else:
                    logger.info("adding all depth images")
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[8]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[8]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[0]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[0]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[9]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[9]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[1]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[1]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[8]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[10]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[0]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[2]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[9]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[11]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[1]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[3]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[12]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[12]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[4]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[4]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[13]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[13]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[5]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[5]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[13]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[14]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[5]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[6]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[12]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[15]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[4]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[7]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
            else:
                logger.info("Only RGB images are added to metadata")
                if perspective=="3d":
                    logger.info("perspective changes are added to metadata")
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[10]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[8]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[11]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[9]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[8]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[10]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[9]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[11]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[15]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[12]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[14]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[13]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[13]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[14]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[12]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[15]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                elif perspective=="2d":
                    logger.info("No perspective changes to metadata")
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[8]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[8]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[9]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[9]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[10]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[10]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[11]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[11]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[12]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[12]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[13]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[13]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[14]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[14]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[15]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[15]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                else:
                    logger.info("adding all depth images")
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[8]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[8]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[9]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[9]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[8]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[10]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[9]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[11]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[12]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[12]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[13]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[13]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[13]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[14]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[12]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[15]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1

    if ROOM == "Kitchen":
        img_number = 0
        for scene in sorted(os.listdir(ROOM_DIR)):
            if any(substring in scene for substring in substrings):
                continue
            scene_buffer = []
            logger.debug("substring: %s", substrings[:-1])
            for img in os.listdir(os.path.join(ROOM_DIR, scene)):
                if any(substring in img for substring in substrings[:-1]):
                    continue
                scene_buffer.append(img)
            scene_buffer.sort()
            if DEPTH:
                logger.info("Adding depth images to metadata")
                if perspective=="2d" or perspective=="3d":
                    logger.info("Adding depth images without perspective changes to metadata")
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[6]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[6]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[0]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[0]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[7]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[7]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[1]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[1]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[8]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[8]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[2]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[2]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[9]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[9]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[3]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[3]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[10]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[10]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[4]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[4]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[11]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[11]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[5]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[5]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                else:
                    logger.info("adding all depth images")
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[6]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[6]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[0]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[0]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[7]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[7]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[1]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[1]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[8]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[8]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[2]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[2]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[9]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[9]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[3]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[3]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[10]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[10]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[4]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[4]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[11]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[11]),
                        "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[5]),
                        "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[5]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
            else:
                logger.info("Only RGB images are added to metadata")
                if perspective=="3d" or perspective=="2d":
                    logger.info("only RGB images with perspective change are added to metadata")
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[6]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[6]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[7]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[7]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[8]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[8]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[9]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[9]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[10]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[10]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[11]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[11]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                else:
                    logger.info("adding all RGB images")
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[6]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[6]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[7]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[7]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[8]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[8]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[9]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[9]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[10]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[10]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1
                    batch.append({
                        "prediction_number": img_number,
                        "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[11]),
                        "image2": os.path.join(ROOM_DIR, scene, scene_buffer[11]),
                        "registration_strategy": f"{registration_strategy}"
                    })
                    img_number += 1  


def get_transformations(ROOM_DIR, logger):
    """
    This function is used to get the transformation matrices for the scenes in the dataset
    """
    logger.info("Adding transformations to metadata")
    for root, dirnames, files in os.walk(ROOM_DIR):
        if "scene" not in root:
            logger.debug("Skipping %s", root)
            continue
        if "ground_truth" in root or "predictions" in root:
            logger.debug("Skipping %s", root)
            continue
        logger.debug(f"root: {root}")
        transformations = utils.load_transformations(
            f"{root}/{root.split('/')[-1]}_transformations.yaml")
        logger.debug(f"transformations: {transformations}")
        for key, value in transformations.items():
            logger.debug("dict value %s", value)
            extrinsics = Extrinsic()
            extrinsics.from_dict(value)
            logger.debug(f"extrinsics: {extrinsics}")
            np.save(f"{root}/{value['file_name'][:-4]}_position.npy", extrinsics.position)
            np.save(f"{root}/{value['file_name'][:-4]}_rotation.npy", extrinsics.rotation)
        intrinsics = Intrinsic()
        intrinsics.from_json(f"data/ObChange/{CAMERA_INFO_JSON_PATH}")
        np.save(f"data/ObChange/{CAMERA_INFO_JSON_PATH[:-5]}.npy", intrinsics.matrix())