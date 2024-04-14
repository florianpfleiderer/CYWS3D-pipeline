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

def prepare_predictions(path_to_preds):
    ''' This function returns the data stored in the .pt files saved after inference in th correct
    format for the mAP function provided by pytorch
    '''
    predictions = torch.load(path_to_preds)
    preds = []
    for i in range(len(predictions)):
        preds.append(dict(
            boxes = predictions[i][0], 
            scores = predictions[i][1][:len(predictions[i][0])],
            labels = torch.zeros(len(predictions[i][0]), dtype=torch.int64)
        ))
    return preds

def prepare_batch(path_to_preds, path_to_gt):
    ''' this function returns the batch prepared for the mAP function provided by pytorch

    Format of Predictions:
    list[tuple(Tensor, Tensor)] where each tuple corresponds to a single image. 
    The first tensor contains the predicted bounding boxes and the second tensor contains the confidence scores. 
    The predicted bounding boxes should be in the format (tl_x, tl_y, br_x, br_y) in absolute image coordinates. 
    The confidence scores should be in the range [0, 1] where 0 means no object and 1 means full confidence that an object is present.
    
    Args:
        path_to_preds (str): path to the predictions file
        path_to_gt (str): path to the ground truth file

    Returns:
        batch (dict): dictionary containing the images and the targets
    '''
    preds = prepare_predictions(path_to_preds)
    ground_truth = torch.load(path_to_gt)
    
    targets = []

    for i in range(len(ground_truth)):
        targets.append(dict(
            boxes = torch.tensor(ground_truth[i][0]),
            labels = torch.zeros(len(ground_truth[i][0]), dtype=torch.int64)
        ))
    return targets, preds

def load_roboflow_export(json_path):
    ''' this function loads the json file exported from roboflow 
    (COCO format = x_min, y_min, width, height) and returns the data in the 
    correct format (x_min,y_min,x_max,y_max)

    Keep in mind that the resulting image should be 224x224 pixels
    '''
    targets = []

    with open(json_path) as file:
        gt = json.load(file)
        for i in range(len(gt['images'])):
            targets.append(dict(
                boxes = torch.tensor([gt['annotations'][i]['bbox'][0], \
                    gt['annotations'][i]['bbox'][1], gt['annotations'][i]['bbox'][0] \
                        + gt['annotations'][i]['bbox'][2], gt['annotations'][i]['bbox'][1] \
                            + gt['annotations'][i]['bbox'][3]], dtype=torch.float32),
                labels = torch.zeros(len(gt['annotations'][i]['bbox']), dtype=torch.int64)
            ))
    return targets

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
        
if __name__ == "__main__":
    targets = load_roboflow_export('annotated_testdata/annotations_coco.json')
    print(targets)
    preds = prepare_predictions('annotated_testdata/batch_image2_predicted_bboxes.pt')
    print(preds)
    print(calculate_mAP(preds, targets))