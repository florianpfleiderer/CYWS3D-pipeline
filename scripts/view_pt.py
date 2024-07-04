#! usr/bin/env python3.9
# Created on Tue Jul 02 2024 by Florian Pfleiderer
# Copyright (c) 2024 TU Wien
"""
This script is used to view pointclopuds from ObChange Dataset.
"""
import torch
import numpy as np
from pprint import pprint
from pprint import pp

CONFIG_FOLDER = "area-400_matching-false_strategy-3d_confidence-03"
FOLDER = "GH30_SmallRoom_05-17_perspective-2d_depth-true_01"


def main(room: str = None):
    

    print("target bboxes:\n")
    target_bboxes = torch.load(f"data/results/{CONFIG_FOLDER}/{FOLDER}/all_target_bboxes.pt")
    pprint(target_bboxes)
    print("\n\npredictions:\n")
    predicted_bboxes = torch.load(f"data/results/{CONFIG_FOLDER}/{FOLDER}/predictions/batch_image2_predicted_bboxes.pt")
    pprint(predicted_bboxes)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)