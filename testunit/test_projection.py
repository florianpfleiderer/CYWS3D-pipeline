import unittest
from src.annotation_pipeline.projection import Intrinsic, Extrinsic

class TestIntrinsic(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.intrinsics = Intrinsic(0,0,0,0,0)

    def test_from_xml(self):
        ''' xml file blablabla 
        '''
        self.intrinsics.from_xml("testunit/testmodel/model1_cameras.xml")
        self.assertAlmostEqual(self.intrinsics.cx, 0.66374072079057678)
        self.assertAlmostEqual(self.intrinsics.cy, -11.513546135775556)
        self.assertAlmostEqual(self.intrinsics.f, 1658.5404555376736)
        self.assertAlmostEqual(self.intrinsics.width, 1920)
        self.assertAlmostEqual(self.intrinsics.height, 1080)
    
class TestExtrinsic(unittest.TestCase):
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