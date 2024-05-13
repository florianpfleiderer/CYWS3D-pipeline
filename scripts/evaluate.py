# Created on Fri Apr 26 2024 by Florian Pfleiderer
# Copyright (c) 2024 TU Wien
"""
Insert Module Description Here
"""
import os
import yaml
import logging
import torch
import json
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from src.evaluation_pipeline import eval_utils, eval_plotter

logging.basicConfig()
logger = logging.getLogger(__name__)

def main(
    room: str = None,
    log_level: str = "INFO"
):
    """
    main function for evaluation pipeline
    """
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.warning("logger set to %s", logger.level)

    # if room is None:
    #     raise ValueError("Please provide a room to evaluate")
    # ROOM_DIR = f"data/GH30_{room}"
    # PREDICTIONS_DIR = ROOM_DIR+"/predictions/batch_image2_predicted_bboxes.pt"
    # TARGET_BBOXES_DIR = ROOM_DIR+"/all_target_bboxes.pt"

    iou_thresholds: np.ndarray = np.arange(0.5, 0.95, 0.05).tolist()
    rec_thresholds: np.ndarray = np.arange(0.1, 1.0, 0.1).tolist()
    max_detection_thresholds: list = [1, 3, 5]

    search_terms = ["perspective-3d", "depth-false"]

    all_preds = []
    all_targets = []

    for folder in os.listdir("data/results"):
        if all(term in folder for term in search_terms):
            logger.info("processing folder: %s", folder)
            preds = torch.load(f"data/results/{folder}/predictions/batch_image2_predicted_bboxes.pt")
            targets = torch.load(f"data/results/{folder}/all_target_bboxes.pt")
        else:  
            continue
        sorted_targets = eval_utils.prepare_target_bboxes(targets, f"data/results/{folder}/input_metadata.yaml")
        all_preds.extend(preds)
        all_targets.extend(sorted_targets)

    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', extended_summary=True, \
        iou_thresholds=iou_thresholds, rec_thresholds=rec_thresholds, \
            max_detection_thresholds=max_detection_thresholds)
    metric.update(all_preds, all_targets)
    mAP = metric.compute()

    eval_utils.map_to_numpy(mAP)

    eval_plotter.plot_precision(mAP, (iou_thresholds, rec_thresholds, max_detection_thresholds), \
        room, f"data/results")

    eval_plotter.plot_recall(mAP, (iou_thresholds, rec_thresholds, max_detection_thresholds), \
        room, f"data/results")

    eval_utils.save_map_as_json(mAP, f"data/results/mAP.json")


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)