# Created on Thu Apr 18 2024 by Florian Pfleiderer
# Copyright (c) 2024 TU Wien
"""
Module for testing the projection module functions.
"""
import unittest
import logging
from src.annotation_pipeline.projection import Intrinsic, Extrinsic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# toggle logging level for different levels of verbosity
logger.setLevel(logging.WARNING)

class TestIntrinsic(unittest.TestCase):
    """ Test the Intrinsic class
    """
    @classmethod
    def setUpClass(self): #TODO; this should say cls instead of self
        self.intrinsics = Intrinsic(0,0,0,0,0)

    def test_from_xml(self):
        ''' xml file blablabla 
        '''
        self.intrinsics.from_xml("testunit/projection/resources/model1_cameras.xml")
        self.assertAlmostEqual(self.intrinsics.cx, 0.66374072079057678)
        self.assertAlmostEqual(self.intrinsics.cy, -11.513546135775556)
        self.assertAlmostEqual(self.intrinsics.f, 1658.5404555376736)
        self.assertAlmostEqual(self.intrinsics.width, 1920)
        self.assertAlmostEqual(self.intrinsics.height, 1080)

class TestExtrinsic(unittest.TestCase):
    """ Test the Extrinsic class
    """
    @classmethod
    def setUpClass(self):
        self.extrinsics = Extrinsic(0,0)

    def test_from_json(self):
        ''' matrix should look like this:
        [[ 0.99393342 -0.01204609  0.1093218   1.19531316]
        [-0.00466678  0.9884695   0.15134815 -1.39707966]
        [-0.10988442 -0.15094017  0.98241665  1.21013233]
        [ 0.          0.          0.          1.        ]]
        '''
        self.extrinsics.from_json("testunit/testmodel/model1_viewpoint_00.json", scale = 10)
        self.assertAlmostEqual(self.extrinsics.extrinsic_matrix[0][0], 0.99393342)
        self.assertAlmostEqual(self.extrinsics.extrinsic_matrix[0][1], -0.01204609)
        self.assertAlmostEqual(self.extrinsics.extrinsic_matrix[0][2], 0.1093218)
        self.assertAlmostEqual(self.extrinsics.extrinsic_matrix[0][3], 1.19531316)
        self.assertAlmostEqual(self.extrinsics.extrinsic_matrix[1][0], -0.00466678)
        self.assertAlmostEqual(self.extrinsics.extrinsic_matrix[1][1], 0.9884695)
        self.assertAlmostEqual(self.extrinsics.extrinsic_matrix[1][2], 0.15134815)
        self.assertAlmostEqual(self.extrinsics.extrinsic_matrix[1][3], -1.39707966)
        self.assertAlmostEqual(self.extrinsics.extrinsic_matrix[2][0], -0.10988442)
        self.assertAlmostEqual(self.extrinsics.extrinsic_matrix[2][1], -0.15094017)
        self.assertAlmostEqual(self.extrinsics.extrinsic_matrix[2][2], 0.98241665)
        self.assertAlmostEqual(self.extrinsics.extrinsic_matrix[2][3], 1.21013233)
        self.assertAlmostEqual(self.extrinsics.extrinsic_matrix[3][0], 0)
        self.assertAlmostEqual(self.extrinsics.extrinsic_matrix[3][1], 0)
        self.assertAlmostEqual(self.extrinsics.extrinsic_matrix[3][2], 0)
        self.assertAlmostEqual(self.extrinsics.extrinsic_matrix[3][3], 1)

    def test_from_yaml(self):
        ''' Output Rotation Matrix should have the following values:
            [  0.9554766,  0.1900868, -0.2256798;
               0.2950667, -0.6155338,  0.7307898;
              -0.0000000, -0.7648432, -0.6442165  ]

        Output Translation Matrix should have the following values:
            x: 1.4447786607505417
            y: 0.8557686332280523
            z: 0.8960933323483593
        '''
        self.extrinsics.from_yaml('testunit/testmodel/transformation.yaml')#
        logger.info(f'position={self.extrinsics.position}')
        logger.info(f'rotation={self.extrinsics.rotation}')
        self.assertAlmostEqual(self.extrinsics.rotation[0][0], 0.95547, places=3)
        self.assertAlmostEqual(self.extrinsics.rotation[1][1], -0.61553, places=3)
        self.assertAlmostEqual(self.extrinsics.rotation[2][2], -0.64421, places=3)
        self.assertAlmostEqual(self.extrinsics.position[0], 1.44477, places=3)
        
        self.assertAlmostEqual(self.extrinsics.homogenous_matrix()[0][0], 0.95547, places=3)
        self.assertAlmostEqual(self.extrinsics.homogenous_matrix()[3][3], 1.0, places=3)
