# Created on Thu Jun 27 2024 by Florian Pfleiderer
# Copyright (c) 2024 TU Wien
"""
This module contains a test suite for the `filter_low_confidence_bboxes` function in the `geometry` module.

The `filter_low_confidence_bboxes` function filters out bounding boxes with low confidence scores based on a given threshold. This test suite covers the basic functionality and edge cases of the `filter_low_confidence_bboxes` function.

Test cases:
- `test_filter_low_confidence_bboxes_filters_out_low_confidence`: Tests if the function correctly filters out bounding boxes with low confidence scores.
- `test_filter_low_confidence_bboxes_keeps_high_confidence`: Tests if the function correctly keeps bounding boxes with high confidence scores.
- `test_filter_low_confidence_bboxes_empty_input`: Tests if the function handles empty input arrays correctly.
- `test_filter_low_confidence_bboxes_all_filtered_out`: Tests if the function correctly handles the case where all bounding boxes are filtered out.
"""
import unittest
import numpy as np
from geometry import filter_low_confidence_bboxes

class TestFilterLowConfidenceBboxes(unittest.TestCase):
    def test_filter_low_confidence_bboxes_filters_out_low_confidence(self):
        bboxes = np.array([[10, 10, 50, 50], [20, 20, 60, 60]])
        scores = np.array([0.1, 0.3])
        threshold = 0.2
        filtered_bboxes, _ = filter_low_confidence_bboxes(bboxes, scores, threshold)
        self.assertEqual(len(filtered_bboxes), 1)
        np.testing.assert_array_equal(filtered_bboxes, np.array([[20, 20, 60, 60]]))

    def test_filter_low_confidence_bboxes_keeps_high_confidence(self):
        bboxes = np.array([[10, 10, 50, 50], [20, 20, 60, 60]])
        scores = np.array([0.3, 0.5])
        threshold = 0.2
        filtered_bboxes, _ = filter_low_confidence_bboxes(bboxes, scores, threshold)
        self.assertEqual(len(filtered_bboxes), 2)

    def test_filter_low_confidence_bboxes_empty_input(self):
        bboxes = np.array([])
        scores = np.array([])
        threshold = 0.2
        filtered_bboxes, _ = filter_low_confidence_bboxes(bboxes, scores, threshold)
        self.assertEqual(len(filtered_bboxes), 0)

    def test_filter_low_confidence_bboxes_all_filtered_out(self):
        bboxes = np.array([[10, 10, 50, 50]])
        scores = np.array([0.1])
        threshold = 0.2
        filtered_bboxes, _ = filter_low_confidence_bboxes(bboxes, scores, threshold)
        self.assertEqual(len(filtered_bboxes), 0)

if __name__ == '__main__':
    unittest.main()
