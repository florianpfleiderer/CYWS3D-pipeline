# Created on Tue Apr 23 2024 by Florian Pfleiderer
# Copyright (c) 2024 TU Wien
"""
This Module creates the input metadata for the inference process.

The script reads images from the specified room (e.g. office) and creates image pairs according
to the following scheme (compare everything to the first (empty scene)): 
- image1: image from scene1
- image2: image from scene2
- depth1: depth image from scene1
- depth2: depth image from scene2
- registration_strategy: 3d
###
- image1: image from scene1
- image2: image from scene3
...
###
- image1: image from scene1
- image2: image from sceneN
...

The the foldername for the specified room is always: 'data/GH30_<Roomname>/' with subfolders
for each scene.

The input_metadata.yaml file is created in the root directory of the room.
"""
import os
import argparse
import yaml
import logging
import numpy as np
from PIL import Image
from src.annotation_pipeline import utils
from src.annotation_pipeline.projection import Intrinsic, Extrinsic
from src.globals \
    import DATASET_FOLDER, IMAGE_FOLDER, ROOM, SCENE, PLANE, PCD_PATH, ANNO_PATH, \
        CAMERA_INFO_JSON_PATH, GT_COLOR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--room", required=True, help = "name of room (e.g. Office)")
parser.add_argument("--depth", required=False, help = "add depth images to metadata")
parser.add_argument("--transformations", required=False, help = "add transformations to metadata")
parser.add_argument("--debug", required=False, help = "set log level to debug")
args = vars(parser.parse_args())

ROOM_DIR = f"data/GH30_{args['room']}"
DEPTH = True if args["depth"] is not None else False
TRANSFORMATIONS = True if args["transformations"] is not None else False
if args["debug"] is not None:
    logger.setLevel(logging.DEBUG)

batch = []

for root, dirnames, files in os.walk(ROOM_DIR):
    if "ground_truth" in root or "predictions" in root:
        logger.debug("Skipping %s", root)
        continue
    logger.debug("%s, %s, files: %s", root, dirnames, files)
    for filename in files:
        if 'depth' in filename and filename.endswith('.png'):
            img = Image.open(os.path.join(root, filename))
            img_gray = img.convert('L')
            img_gray.save(os.path.join(root, filename))
            logger.info("Converted %s in %s to greyscale", filename, root)

logger.info("Converted all depth images to greyscale")

# for root, dirnames, files in os.walk(ROOM_DIR):
#     if "scene" not in root:
#         logger.debug("Skipping %s", root)
#         continue
#     if "ground_truth" in root or "predictions" in root:
#         logger.debug("Skipping %s", root)
#         continue
#     logger.debug(f"root: {root}")
#     transformations = utils.load_transformations(f"{root}/{root.split('/')[-1]}_transformations.yaml")
#     logger.debug(f"transformations: {transformations}")
#     for key, value in transformations.items():
#         logger.debug("dict value %s", value)
#         extrinsics = Extrinsic()
#         extrinsics.from_dict(value)
#         logger.debug(f"extrinsics: {extrinsics}")
#         np.save(f"{root}/{value['file_name'][:-4]}_position.npy", extrinsics.position)
#         np.save(f"{root}/{value['file_name'][:-4]}_rotation.npy", extrinsics.rotation)
#     intrinsics = Intrinsic()
#     intrinsics.from_json(f"data/ObChange/{CAMERA_INFO_JSON_PATH}")
#     np.save(f"data/ObChange/{CAMERA_INFO_JSON_PATH[:-5]}.npy", intrinsics.matrix())

scene1_buffer = sorted([f for f in os.listdir(os.path.join(ROOM_DIR, "scene1")) \
    if (".png" in f)])

# transformations = None

# for i in range(6):
#     for j in range(i, 6):
#         if i == j:
#             continue
#         transformations_1 = utils.load_transformations(os.path.join(ROOM_DIR, f"scene{i+1}", f"scene{i+1}_transformations.yaml"))
#         logger.debug(f"transformations: {transformations_1}")
#         transformations_2 = utils.load_transformations(os.path.join(ROOM_DIR, f"scene{j+1}", f"scene{j+1}_transformations.yaml"))
#         logger.debug(f"transformations: {transformations_2}")
#         for p in range(len(transformations_1)):
#             batch.append(dict(
#                 image1=os.path.join(ROOM_DIR, f"scene{i+1}", transformations_1[p]["file_name"]),
#                 image2=os.path.join(ROOM_DIR, f"scene{j+1}", transformations_2[p]["file_name"]),
#                 registration_strategy="3d"
#             ))

for scene in sorted(os.listdir(ROOM_DIR)):
    if "predictions" in scene or "scene1" in scene or "yaml" in scene or ".pt" in scene:
        continue
    scene_buffer = []
    for img in os.listdir(os.path.join(ROOM_DIR, scene)):
        if ".yaml" in img or "ground_truth" in img or ".npy" in img:
            continue
        scene_buffer.append(img)
    scene_buffer.sort()
    spacer = len(scene_buffer)//2
    if DEPTH:
        logger.info("Adding depth images to metadata")
        for i in range(spacer//2):
            batch.append({
                "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[4]),
                "image2": os.path.join(ROOM_DIR, scene, scene_buffer[spacer+i]),
                "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[0]),
                "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[i]),
                "registration_strategy": "3d"
            })
        for i in range(spacer//2):
            batch.append({
                "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[6]),
                "image2": os.path.join(ROOM_DIR, scene, scene_buffer[spacer+2+i]),
                "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[2]),
                "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[2+i]),
                "registration_strategy": "3d"
            })
    elif TRANSFORMATIONS:
        logger.info("Adding transformations to metadata")
        for i in range(spacer//2):
            batch.append({
                "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[4]),
                "image2": os.path.join(ROOM_DIR, scene, scene_buffer[spacer+i]),
                "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[0]),
                "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[i]),
                "intrinsics1": DATASET_FOLDER+CAMERA_INFO_JSON_PATH[:-5]+".npy",
                "intrinsics2": DATASET_FOLDER+CAMERA_INFO_JSON_PATH[:-5]+".npy",
                "position1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[4][:-4]+"_position.npy"),
                "position2": os.path.join(ROOM_DIR, scene, scene_buffer[spacer+i][:-4]+"_position.npy"),
                "rotation1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[4][:-4]+"_rotation.npy"),
                "rotation2": os.path.join(ROOM_DIR, scene, scene_buffer[spacer+i][:-4]+"_rotation.npy"),
                "registration_strategy": "3d"
            })
        for i in range(spacer//2):
            batch.append({
                "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[6]),
                "image2": os.path.join(ROOM_DIR, scene, scene_buffer[spacer+2+i]),
                "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[2]),
                "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[2+i]),
                "intrinsics1": DATASET_FOLDER+CAMERA_INFO_JSON_PATH[:-5]+".npy",
                "intrinsics2": DATASET_FOLDER+CAMERA_INFO_JSON_PATH[:-5]+".npy",
                "position1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[6][:-4]+"_position.npy"),
                "position2": os.path.join(ROOM_DIR, scene, scene_buffer[spacer+2+i][:-4]+"_position.npy"),
                "rotation1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[6][:-4]+"_rotation.npy"),
                "rotation2": os.path.join(ROOM_DIR, scene, scene_buffer[spacer+2+i][:-4]+"_rotation.npy"),
                "registration_strategy": "3d"
            })
    else:
        logger.info("Only RGB images are added to metadata")
        for i in range(spacer//2):
            batch.append({
                "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[4]),
                "image2": os.path.join(ROOM_DIR, scene, scene_buffer[spacer+i]),
                "registration_strategy": "3d"
            })
        for i in range(spacer//2):
            batch.append({
                "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[6]),
                "image2": os.path.join(ROOM_DIR, scene, scene_buffer[spacer+2+i]),
                "registration_strategy": "3d"
            })

with open(os.path.join(ROOM_DIR, "input_metadata.yaml"), "w") as f:
    yaml.safe_dump({"batch": batch}, f)
    
