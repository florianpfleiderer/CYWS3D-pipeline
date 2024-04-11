''' This modules provides for calculating the mean average precision (mAP) of the predictions
made by the cyws3d on th eobchange dataset (specifically on the selected RGB Frames). 

What data do we need ?
annotated images with confidence scores
annotated ground truth images

first implementation just for 10 pictures and 10 annotations in a folder

'''

import torch
from collections import Counter
# from iou import intersection_over_union7

def calculate_mAP(pred_boxes: list, 
                    true_boxes:list, 
                    iou_threshold=0.5,
                    box_format='corners', 
                    num_classes=20):
    ''' used to calculate mAP on given bboxes.

    Args:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar to pred_boxes except all the correct ones
        iou_threshold (float, optional): Defaults to 0.5.
        box_format (str, optional): Defaults to 'corners'.
        num_classes (int, optional): Defaults to 20.
    '''
    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):
        detections =[]
        ground_truths = []
        
if __name__ == "__main__":
    img01 = torch.load('annotated_testdata/batch_image1_predicted_bboxes.pt')
    img02 = torch.load('annotated_testdata/batch_image2_predicted_bboxes.pt')
    torch.save(img01[0], 'annotated_testdata/batch_image1_predicted_data.pt')
    torch.save(img02[0], 'annotated_testdata/batch_image2_predicted_data.pt')
    torch.save(img01[1], 'annotated_testdata/batch_image1_gt.pt')
    torch.save(img02[1], 'annotated_testdata/batch_image2_gt.pt')
 