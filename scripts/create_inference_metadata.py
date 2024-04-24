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
from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("--room", required=True, help = "name of room (e.g. Office)")
args = vars(parser.parse_args())
ROOM_DIR = f"data/GH30_{args['room']}"

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

scene1_buffer = sorted([f for f in os.listdir(os.path.join(ROOM_DIR, "scene1")) \
    if (".yaml" not in f and "ground_truth" not in f)])

for scene in sorted(os.listdir(ROOM_DIR)):
    if "predictions" in scene or "scene1" in scene or "yaml" in scene:
        continue
    scene_buffer = []
    # TODO: safe transformations as numpy array in ros_playground
    # transformations = yaml.safe_load(open(os.path.join(ROOM_DIR, scene, f"{scene}_transformations.yaml")))
    # logger.debug("Transformations: %s", transformations)

    for img in os.listdir(os.path.join(ROOM_DIR, scene)):
        if ".yaml" in img or "ground_truth" in img: 
            continue
        scene_buffer.append(img)
    scene_buffer.sort()
    spacer = len(scene_buffer)//2
    for i in range(spacer//2):
        batch.append({
            "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[4]),
            "image2": os.path.join(ROOM_DIR, scene, scene_buffer[spacer+i]),
            "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[0]),
            "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[i]),
            "registration_strategy": "3d"
        })
        batch.append({
            "image1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[6]),
            "image2": os.path.join(ROOM_DIR, scene, scene_buffer[spacer+2+i]),
            "depth1": os.path.join(ROOM_DIR, "scene1", scene1_buffer[2]),
            "depth2": os.path.join(ROOM_DIR, scene, scene_buffer[2+i]),
            "registration_strategy": "3d"
        })

with open(os.path.join(ROOM_DIR, "input_metadata.yaml"), "w") as f:
    yaml.safe_dump({"batch": batch}, f)
    
