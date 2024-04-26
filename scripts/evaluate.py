# Created on Fri Apr 26 2024 by Florian Pfleiderer
# Copyright (c) 2024 TU Wien
"""
Insert Module Description Here
"""
import os
import argparse
import yaml
import logging
import torch
from src.evaluation_pipeline.calculate_mAP import calculate_mAP

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--room", required=True, help = "name of room (e.g. Office)")
args = vars(parser.parse_args())
ROOM_DIR = f"data/GH30_{args['room']}"
PREDICTIONS_DIR = ROOM_DIR+"/predictions/batch_image2_predicted_bboxes.pt"
TARGET_BBOXES_DIR = ROOM_DIR+"/all_target_bboxes.pt"
preds = torch.load(PREDICTIONS_DIR)
targets = torch.load(TARGET_BBOXES_DIR)

mAP = calculate_mAP(preds, targets)
print(f"mAP: {mAP['map']}")