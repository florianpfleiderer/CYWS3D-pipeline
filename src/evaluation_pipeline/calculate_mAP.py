''' This modules provides for calculating the mean average precision (mAP) of the predictions
made by the cyws3d on th eobchange dataset (specifically on the selected RGB Frames). 

What data do we need ?
annotated images with confidence scores
annotated ground truth images

first implementation just for 10 pictures and 10 targets in a folder

'''

import json
import torch
from torchmetrics.detection import MeanAveragePrecision

def calculate_mAP(preds, targets):
    ''' this function calculates the mean average precision of the predictions made by the model

    The input data should be in the following format:
    A list consisting of dictionaries each containing the key-values (each dictionary corresponds to a single image). Parameters that should be provided per dict:
        boxes (Tensor): float tensor of shape (num_boxes, 4) containing num_boxes detection boxes of the format specified in the constructor. By default, this method expects (xmin, ymin, xmax, ymax) in absolute image coordinates, but can be changed using the box_format parameter. Only required when iou_type=”bbox”.
        scores (Tensor): float tensor of shape (num_boxes) containing detection scores for the boxes.
        labels (Tensor): integer tensor of shape (num_boxes) containing 0-indexed detection classes for the boxes.

    Args:
        targets (list): list of dictionaries containing the ground truth bounding boxes
        preds (list): list of dictionaries containing the predicted bounding boxes

    Returns:
        mAP (float): mean average precision of the model
    '''
    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
    metric.update(preds, targets)
    return metric.compute()
