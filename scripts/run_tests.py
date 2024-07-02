#! usr/bin/env python3.9
# Created on Thu Apr 18 2024 by Florian Pfleiderer
# Copyright (c) 2024 TU Wien
"""
Runs all tests defined in the unittest/ folder.
"""

import unittest
from testunit import test_projection, test_evaluation

testsuite = unittest.TestLoader().loadTestsFromModule(test_projection)
testsuite.addTests(unittest.TestLoader().loadTestsFromModule(test_evaluation))
runner = unittest.TextTestRunner()
runner.run(testsuite)
