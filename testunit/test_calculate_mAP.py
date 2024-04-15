import unittest
import torch
from src.evaluation_pipeline import calculate_mAP

batch_img2 = torch.load('annotated_testdata/batch_image2_predicted_bboxes.pt')
print(batch_img2)