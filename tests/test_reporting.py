"""
Unit tests for reporting and export module.
"""

import os
import tempfile
import unittest

from src.reporting import ReportGenerator


class TestReporting(unittest.TestCase):

    def test_calculate_summary(self):
        panels = [{"id": f"P_{i}"} for i in range(12)]
        defects = [
            {"panel_id": "P_1", "severity": "CRITICAL", "est_power_loss_pct": 20.0},
            {"panel_id": "P_2", "severity": "WARNING", "est_power_loss_pct": 8.0}
        ]
        summary = ReportGenerator.calculate_summary(panels, defects)

        self.assertEqual(summary["total_panels"], 12)
        self.assertEqual(summary["defective_panels"], 2)
        self.assertEqual(summary["healthy_panels"], 10)
        self.assertEqual(summary["critical_defects"], 1)
        self.assertEqual(summary["warning_defects"], 1)
        self.assertGreater(summary["total_power_loss_kw"], 0)

    def test_export_csv(self):
        defects = [
            {
                "id": "DEF-001",
                "image_name": "panel_01.jpg",
                "panel_id": "PV-R01-M02",
                "type": "HOTSPOT",
                "type_name": "Localized Hotspot",
                "severity": "CRITICAL",
                "confidence": 0.95,
                "delta_t_celsius": 14.5,
                "est_power_loss_pct": 18.5,
                "bbox_global": (100, 150, 40, 40)
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "inspection_results.csv")
            ReportGenerator.export_csv(csv_path, defects)
            self.assertTrue(os.path.exists(csv_path))
            self.assertGreater(os.path.getsize(csv_path), 0)

    def test_export_html_and_pdf(self):
        panels = [{"id": f"P_{i}"} for i in range(12)]
        defects = [
            {
                "id": "DEF-001",
                "panel_id": "PV-R01-M02",
                "type": "HOTSPOT",
                "type_name": "Localized Hotspot",
                "severity": "CRITICAL",
                "confidence": 0.95,
                "delta_t_celsius": 14.5,
                "est_power_loss_pct": 18.5,
                "bbox_global": (100, 150, 40, 40)
            }
        ]
        summary = ReportGenerator.calculate_summary(panels, defects)

        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = os.path.join(tmpdir, "report.html")
            ReportGenerator.export_html(html_path, summary, defects)
            self.assertTrue(os.path.exists(html_path))
            self.assertGreater(os.path.getsize(html_path), 0)

            pdf_path = os.path.join(tmpdir, "report.pdf")
            ReportGenerator.export_pdf(pdf_path, summary, defects)
            self.assertTrue(os.path.exists(pdf_path))
            self.assertGreater(os.path.getsize(pdf_path), 0)


if __name__ == "__main__":
    unittest.main()

