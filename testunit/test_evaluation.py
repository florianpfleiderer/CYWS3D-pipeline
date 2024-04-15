import unittest
import torch
from src.evaluation_pipeline.calculate_mAP import calculate_mAP
from src.evaluation_pipeline.obchange_dataloader import load_roboflow_export

class TestCalculate_mAP(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.predictions = torch.load('testunit/annotated_testdata/batch_image2_predicted_bboxes.pt')
        self.targets = load_roboflow_export('testunit/annotated_testdata/annotations_coco.json')

    def test_calculate_mAP(self):
        ''' test if the mAP is calculated correctly
        '''
        # print(f'predicted bboxes: {self.predictions}')
        # print(f'target bboxes: {self.targets}')
        self.map = calculate_mAP(self.predictions, self.targets)['map']
        print(f'mAP: {self.map}')
        self.assertGreaterEqual(self.map, 0.5)

