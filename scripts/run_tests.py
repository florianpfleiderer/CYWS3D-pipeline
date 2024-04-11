import unittest
from testunit import test_projection

testsuite = unittest.TestLoader().loadTestsFromModule(test_projection)
runner = unittest.TextTestRunner()
runner.run(testsuite)
