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
    path: str = "data/results",
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

    iou_thresholds: np.ndarray = np.around(np.arange(0.5, 0.95, 0.05), 2).tolist()
    rec_thresholds: np.ndarray = np.around(np.arange(0, 1.0, 0.01), 1).tolist()
    max_detection_thresholds: list = [1, 10, 100]

    search_terms = []

    all_preds = []
    all_targets = []

    # Wipe the metrics.yaml file
    with open("data/results/metrics.yaml", "w") as file:
        file.write("")

    for config_folder in sorted(os.listdir(path)):
        if ".DS" in config_folder or "OLD" in config_folder:
            continue
        logger.info("processing config folder: %s", config_folder)
        for folder in sorted(os.listdir(f"{path}/{config_folder}")):
            if ".DS" in folder:
                continue
            if all(term in folder for term in search_terms):
                logger.info("processing folder: %s", folder)
                preds = torch.load(f"{path}/{config_folder}/{folder}/predictions/batch_image2_predicted_bboxes.pt")
                targets = torch.load(f"{path}/{config_folder}/{folder}/all_target_bboxes.pt")
            else:  
                continue
            sorted_targets = eval_utils.prepare_target_bboxes(targets, f"{path}/{config_folder}/{folder}/input_metadata.yaml")
            all_preds.extend(preds)
            all_targets.extend(sorted_targets)

        metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', extended_summary=True, \
            iou_thresholds=iou_thresholds, rec_thresholds=rec_thresholds, \
                max_detection_thresholds=max_detection_thresholds)
        metric.update(all_preds, all_targets)
        mAP = metric.compute()
        # print(len(all_targets))

        eval_utils.map_to_numpy(mAP)
        print(isinstance(mAP["precision"], np.ndarray))
        print(isinstance(mAP["recall"], np.ndarray))

        # Plot precision-recall curve for large objects
        plt.figure(figsize=(10, 6))
        for idx, iou_threshold in enumerate(iou_thresholds):
            recall = mAP["recall"][idx, 0, 0]
            precision = mAP["precision"][idx, 0, 0, 2]
            plt.plot(recall, label=f'IoU={iou_threshold:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve for all Objects')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"data/results/precision_recall_curve.png")

        eval_plotter.plot_precision(mAP, (iou_thresholds, rec_thresholds, max_detection_thresholds), \
            room, f"data/results")
        eval_plotter.plot_recall(mAP, (iou_thresholds, rec_thresholds, max_detection_thresholds), \
            room, f"data/results")
        # eval_plotter.plot_ious(mAP, room, f"data/results")

        # Create or open the YAML file in append mode
        with open("data/results/metrics.yaml", "a") as file:
            # Write the mAP, precision, and recall to the file
            yaml.dump({
                config_folder: {
                    "mAP": float(mAP["map_50"]),
                    "precision": float(mAP["precision"][1, 0, 0, 0]),
                    "recall": float(mAP["recall"][1, 0, 0])
                }
            }, file)
            logger.info("mAP written to file")

        eval_utils.save_map_as_json(mAP, f"data/results/mAP.json")

if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)