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