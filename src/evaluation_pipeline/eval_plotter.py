# Created on Wed May 08 2024 by Florian Pfleiderer
# Copyright (c) 2024 TU Wien
"""
Helper functions to plot results from torchmetrics.MeanaveragePrecision
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_precision(mAP, thresholds, room_name, room_path):
    """
    Plots the precision values for different configurations of the MeanAveragePrecision class

    Args:
        mAP (dict): dictionary containing the precision values
        thresholds (tuple): tuple containing the iou_thresholds, rec_thresholds and max_detection_thresholds
        room_name (str): name of the room
        room_path (str): path to the room directory

    Returns:
        None
    """
    iou_thresholds, rec_thresholds, max_detection_thresholds = thresholds
    fig1, axs = plt.subplots(3, 1, figsize=(10, 10))
    fig1.suptitle(f"precision for {room_name}")

    # precision vs IoU at different bbox sizes
    areas = ['all', 'small', 'medium', 'large']
    colors = ['blue', 'green', 'orange', 'red']

    for i, area in enumerate(areas):
        precision_values = mAP['precision'][:, 6, i, 0]
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
        precision_values = mAP['precision'][:, 6, 0, i]
        precision_values = [p if p > 0 else 0 for p in precision_values]
        axs[2].plot(iou_thresholds, precision_values, label=max_det)

    axs[2].set_xlabel('IoU Threshold')
    axs[2].set_ylabel('Precision')
    axs[2].set_title('Precision vs IoU Threshold for different max detection thresholds')
    axs[2].legend()

    plt.tight_layout()
    fig1.savefig(f"{room_path}/precision_{room_name}.png")

def plot_recall(mAP, thresholds, room_name, room_path):
    """
    Plots the recall values for different configurations of the MeanAveragePrecision class

    Args:
        mAP (dict): dictionary containing the recall values
        thresholds (tuple): tuple containing the iou_thresholds, rec_thresholds and max_detection_thresholds
        room_name (str): name of the room
        room_path (str): path to the room directory

    Returns:
        None
    """
    iou_thresholds, rec_thresholds, max_detection_thresholds = thresholds
    fig2, axs = plt.subplots(3, 1, figsize=(10, 10))
    fig2.suptitle(f"recall for {room_name}")

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
    fig2.savefig(f"{room_path}/recall_{room_name}.png")

def plot_precision_recall_curve(preds, targets, path):
    """
    This should plot the precision recall curve for the given predictions and targets.
    """
    pass

def plot_ious(ious, room_name, room_path):
    """
    This should plot the IOU values for each image over the image number in the dataset.

    The Iou Values are given as follows:
    "ious": {
    "0": [],
    "1": [
      [
        0.6484149694442749,
        0.0,
        0.0
      ],
      [
        0.0,
        0.0,
        0.5454545617103577
      ],
      [
        0.0,
        0.5468354225158691,
        0.0
      ]
    ],
    "2": [
      0.0,
      0.0,
      0.6436781883239746
    ], ...
    where the first index is the image number and Each value is a tensor with shape (n,m) 
    where n is the number of detections and m is the number of ground truth boxes 
    for that image/class combination.
    """
    fig3, ax = plt.subplots(figsize=(10, 6))
    fig3.suptitle("IOU values per image")
    
    image_numbers = list(ious.keys())
    average_iou_values = [np.mean(ious[image]) for image in image_numbers]
    
    ax.plot(image_numbers, average_iou_values, marker='o')
    ax.set_xlabel('Image Number')
    ax.set_ylabel('Average IOU')
    ax.set_title('Average IOU values per image')
    
    plt.tight_layout()
    fig3.savefig(f"{room_path}/recall_{room_name}.png")
