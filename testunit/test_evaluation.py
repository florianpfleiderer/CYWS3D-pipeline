import unittest
import logging
import torch
from src.evaluation_pipeline.calculate_mAP import calculate_mAP
from src.evaluation_pipeline.obchange_dataloader import load_pascal_voc_export

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# toggle logging level for different levels of verbosity
logger.setLevel(logging.WARNING)

class TestCalculate_mAP(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.predictions = torch.load('testunit/annotated_testdata/batch_image2_predicted_bboxes.pt')
        self.targets = load_pascal_voc_export('testunit/annotated_testdata/test')

    def test_calculate_mAP(self):
        ''' test if the mAP is calculated correctly
        '''
        logger.info(f'predicted bboxes: {self.predictions}')
        logger.info(f'target bboxes: {self.targets}')
        self.map = calculate_mAP(self.predictions, self.targets)
        logger.debug(f'mAP: {self.map}')
        self.assertGreaterEqual(self.map['map'], 0.5)

