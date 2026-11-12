"""
Unit tests for anomaly detection engine.
"""

import os
import unittest
import numpy as np

from src.preprocessing import ImagePreprocessor, PanelSegmenter
from src.detection import AnomalyDetector, DEFECT_CLASSES


class TestDetection(unittest.TestCase):

    def test_thermal_anomalies_detection(self):
        img_path = "data/original/panel_01.jpg"
        self.assertTrue(os.path.exists(img_path))

        img = ImagePreprocessor.load_image(img_path)
        enhanced = ImagePreprocessor.apply_clahe(img)
        segmenter = PanelSegmenter()
        panels, _ = segmenter.segment_panels(enhanced)

        detector = AnomalyDetector()
        defects = detector.analyze_panels(panels, modality="THERMAL_IR")

        self.assertGreater(len(defects), 0)
        defect_types = set(d["type"] for d in defects)
        # Check that thermal hotspot or bypass diode failure is identified
        self.assertTrue(("HOTSPOT" in defect_types) or ("BYPASS_DIODE_FAILURE" in defect_types))

    def test_optical_defects_detection(self):
        img_path = "data/original/panel_02.jpg"
        self.assertTrue(os.path.exists(img_path))

        img = ImagePreprocessor.load_image(img_path)
        enhanced = ImagePreprocessor.apply_clahe(img)
        segmenter = PanelSegmenter()
        panels, _ = segmenter.segment_panels(enhanced)

        detector = AnomalyDetector()
        defects = detector.analyze_panels(panels, modality="RGB_OPTICAL")

        self.assertGreater(len(defects), 0)
        for d in defects:
            self.assertIn(d["type"], DEFECT_CLASSES)
            self.assertIn(d["severity"], ["CRITICAL", "WARNING", "ADVISORY"])


if __name__ == "__main__":
    unittest.main()

