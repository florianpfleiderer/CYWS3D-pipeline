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
import json
from torchmetrics.detection import MeanAveragePrecision
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

metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', extended_summary=True, \
    max_detection_thresholds=[1, 3, 5])
metric.update(preds, targets)
mAP = metric.compute()

mAP_data = {}
mAP_data_per_image = []

for key, value in mAP.items():
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            mAP[key] = value.item()
        else:
            mAP[key] = value.tolist()
        mAP_data.update({key: mAP[key]})
    print(f"{key}: {mAP[key]}\n\n#######################################")

for i in range(len(preds)):
    iou = mAP['ious'][i].tolist()
    precision = mAP['precisions']

print(f"mAP: {mAP['map']}")
print(f"mAP_50: {mAP['map_50']}")
print(f"mAP_75: {mAP['map_75']}")
print(f"mAP_small: {mAP['map_small']}")
print(f"mAP_medium: {mAP['map_medium']}")
print(f"mAP_large: {mAP['map_large']}")

print(f"mAR_1: {mAP['mar_1']}")
print(f"mAR_3: {mAP['mar_3']}")
print(f"mAR_5: {mAP['mar_5']}")
print(f"mAR_small: {mAP['mar_small']}")
print(f"mAR_medium: {mAP['mar_medium']}")
print(f"mAR_large: {mAP['mar_large']}")

with open(ROOM_DIR+"/predictions/mAP.yaml", "w") as f:
    yaml.safe_dump(mAP_data, f)
