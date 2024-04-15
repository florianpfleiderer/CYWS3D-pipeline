import unittest
from testunit import test_projection, test_evaluation

testsuite = unittest.TestLoader().loadTestsFromModule(test_projection)
testsuite.addTests(unittest.TestLoader().loadTestsFromModule(test_evaluation))
runner = unittest.TextTestRunner()
runner.run(testsuite)
