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
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from src.evaluation_pipeline.calculate_mAP import calculate_mAP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--room", required=True, help = "name of room (e.g. Office)")
args = vars(parser.parse_args())
ROOM_DIR = f"data/GH30_{args['room']}"
PREDICTIONS_DIR = ROOM_DIR+"/predictions/batch_image2_predicted_bboxes.pt"
TARGET_BBOXES_DIR = ROOM_DIR+"/all_target_bboxes.pt"
preds = torch.load(PREDICTIONS_DIR)
targets = torch.load(TARGET_BBOXES_DIR)
iou_thresholds = np.arange(0.3, 0.95, 0.05).tolist()
rec_thresholds = np.arange(0.1, 1.0, 0.1).tolist()
max_detection_thresholds = [1, 3, 5]

metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', extended_summary=True, \
    iou_thresholds=iou_thresholds, rec_thresholds=rec_thresholds, \
        max_detection_thresholds=max_detection_thresholds)
metric.update(preds, targets)
mAP = metric.compute()

for key, value in mAP.items():
    if isinstance(value, torch.Tensor):
        mAP[key] = value.squeeze().cpu().numpy()

fig1, axs = plt.subplots(3, 1, figsize=(10, 10))
fig1.suptitle(f"precision for {args['room']}")

# precision vs IoU at different bbox sizes
areas = ['all', 'small', 'medium', 'large']
colors = ['blue', 'green', 'orange', 'red']

for i, area in enumerate(areas):
    precision_values = mAP['precision'][:, 0, i, 0]
    # print(precision_values)
    precision_values = [p if p > 0 else 0 for p in precision_values]
    axs[0].plot(iou_thresholds, precision_values, label=area, color=colors[i])

axs[0].set_xlabel('IoU Threshold')
axs[0].set_ylabel('Precision')
axs[0].set_title('Precision vs IoU Threshold for different bbox sizes')
axs[0].legend()

# precision vs IoU at different recall thresholds
for i, rec in enumerate(rec_thresholds):
    precision_values = mAP['precision'][:, i, 0, 0]
    precision_values = [p if p > 0 else 0 for p in precision_values]
    axs[1].plot(iou_thresholds, precision_values, label=rec)

axs[1].set_xlabel('IoU Threshold')
axs[1].set_ylabel('Precision')
axs[1].set_title('Precision vs IoU Threshold for different recall thresholds')
axs[1].legend()

# precision over IoU at different max detection thresholds
for i, max_det in enumerate(max_detection_thresholds):
    precision_values = mAP['precision'][:, 0, 0, i]
    precision_values = [p if p > 0 else 0 for p in precision_values]
    axs[2].plot(iou_thresholds, precision_values, label=max_det)

axs[2].set_xlabel('IoU Threshold')
axs[2].set_ylabel('Precision')
axs[2].set_title('Precision vs IoU Threshold for different max detection thresholds')
axs[2].legend()

plt.tight_layout()
fig1.savefig(f"{ROOM_DIR}/precision_{args['room']}.png")

fig2, axs = plt.subplots(3, 1, figsize=(10, 10))
fig2.suptitle(f"recall for {args['room']}")

# recall vs IoU at different bbox sizes
areas = ['all', 'small', 'medium', 'large']
colors = ['blue', 'green', 'orange', 'red']

for i, area in enumerate(areas):
    recall_values = mAP['recall'][:, i, 0]
    recall_values = [r if r > 0 else 0 for r in recall_values]
    axs[0].plot(iou_thresholds, recall_values, label=area, color=colors[i])

axs[0].set_xlabel('IoU Threshold')
axs[0].set_ylabel('Recall')
axs[0].set_title('Recall vs IoU Threshold for different bbox sizes')
axs[0].legend()

# recall vs IoU at different max detection thresholds
for i, max_det in enumerate(max_detection_thresholds):
    recall_values = mAP['recall'][:, 0, i]
    recall_values = [r if r > 0 else 0 for r in recall_values]
    axs[1].plot(iou_thresholds, recall_values, label=max_det)

axs[1].set_xlabel('IoU Threshold')
axs[1].set_ylabel('Recall')
axs[1].set_title('Recall vs IoU Threshold for different max detection thresholds')
axs[1].legend()

plt.tight_layout()
fig2.savefig(f"{ROOM_DIR}/recall_{args['room']}.png")

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

