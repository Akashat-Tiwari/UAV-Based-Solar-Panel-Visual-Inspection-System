"""
Unit tests for image preprocessing and panel segmentation.
"""

import os
import unittest
import numpy as np
import cv2

from src.preprocessing import ImagePreprocessor, PanelSegmenter


class TestPreprocessing(unittest.TestCase):

    def test_load_image(self):
        sample_path = "data/original/panel_01.jpg"
        self.assertTrue(os.path.exists(sample_path))
        img = ImagePreprocessor.load_image(sample_path)
        self.assertIsInstance(img, np.ndarray)
        self.assertTrue(img.shape[0] > 0 and img.shape[1] > 0)

    def test_detect_modality(self):
        img_thermal = ImagePreprocessor.load_image("data/original/panel_01.jpg")
        mod_thermal = ImagePreprocessor.detect_modality(img_thermal, filename="panel_01.jpg")
        self.assertIn("THERMAL", mod_thermal)

        img_rgb = ImagePreprocessor.load_image("data/original/panel_02.jpg")
        mod_rgb = ImagePreprocessor.detect_modality(img_rgb, filename="panel_02.jpg")
        self.assertEqual(mod_rgb, "RGB_OPTICAL")

    def test_apply_clahe(self):
        img = ImagePreprocessor.load_image("data/original/panel_02.jpg")
        enhanced = ImagePreprocessor.apply_clahe(img, clip_limit=3.0)
        self.assertEqual(enhanced.shape, img.shape)
        self.assertIsInstance(enhanced, np.ndarray)

    def test_panel_segmenter(self):
        img = ImagePreprocessor.load_image("data/original/panel_01.jpg")
        segmenter = PanelSegmenter()
        panels, mask = segmenter.segment_panels(img)

        self.assertGreaterEqual(len(panels), 4)
        self.assertEqual(mask.shape, img.shape[:2])
        first_panel = panels[0]
        self.assertIn("id", first_panel)
        self.assertIn("bbox", first_panel)
        self.assertIn("roi", first_panel)


if __name__ == "__main__":
    unittest.main()

