"""
Photovoltaic Anomaly Detection Engine.
Detects Thermal Hotspots, Bypass Diode Failures, Micro-Cracks, and Surface Soiling.
"""

from typing import List, Dict, Any, Optional
import cv2
import numpy as np

DEFECT_CLASSES = {
    "HOTSPOT": {
        "name": "Localized Hotspot",
        "description": "Severely overheated cell area caused by internal shunt or localized shading.",
        "color_bgr": (0, 0, 255),       # Red
        "color_hex": "#EF4444",
        "default_severity": "CRITICAL",
        "avg_power_loss_pct": 18.5,
    },
    "BYPASS_DIODE_FAILURE": {
        "name": "Bypass Diode Failure / Sub-string Defect",
        "description": "Failure causing 1/3 or 2/3 of the panel to overheat or remain open-circuit.",
        "color_bgr": (0, 140, 255),     # Orange
        "color_hex": "#F97316",
        "default_severity": "CRITICAL",
        "avg_power_loss_pct": 33.3,
    },
    "SURFACE_CRACK": {
        "name": "Micro-Crack / Fracture",
        "description": "Physical glass or silicon cell fracture, posing long-term water ingress risk.",
        "color_bgr": (255, 0, 255),     # Magenta
        "color_hex": "#D946EF",
        "default_severity": "WARNING",
        "avg_power_loss_pct": 8.0,
    },
    "SOILING_DUST": {
        "name": "Soiling & Dust Accumulation",
        "description": "Accumulation of airborne particulate, sand, or bird droppings reducing irradiance.",
        "color_bgr": (0, 215, 255),     # Yellow-Gold
        "color_hex": "#EAB308",
        "default_severity": "ADVISORY",
        "avg_power_loss_pct": 5.5,
    }
}

SEVERITY_LEVELS = {
    "CRITICAL": {"priority": 1, "badge_color": "#EF4444", "action": "Immediate replacement / electrical isolation required within 48h"},
    "WARNING": {"priority": 2, "badge_color": "#F59E0B", "action": "Schedule on-site technician maintenance within 14 days"},
    "ADVISORY": {"priority": 3, "badge_color": "#3B82F6", "action": "Log for scheduled routine cleaning / seasonal inspection"}
}


class AnomalyDetector:
    """Multi-anomaly detection engine for aerial solar plant inspections."""

    def __init__(self, hotspot_delta_thresh: float = 6.0):
        self.hotspot_delta_thresh = hotspot_delta_thresh

    def analyze_panels(self, panels: List[Dict[str, Any]], modality: str = "RGB_OPTICAL") -> List[Dict[str, Any]]:
        """
        Executes defect detection across all segmented solar panels.
        Returns a list of identified anomalies with metadata and bounding boxes.
        """
        all_defects = []
        is_thermal = "THERMAL" in modality.upper()

        for panel in panels:
            roi = panel["roi"]
            px, py, pw, ph = panel["bbox"]
            panel_id = panel["id"]

            if roi is None or pw < 30 or ph < 30:
                continue

            # Crop inner 85% to isolate active silicon cells from cold aluminum frames
            pad_y, pad_x = max(2, int(ph * 0.08)), max(2, int(pw * 0.08))
            inner = roi[pad_y:ph-pad_y, pad_x:pw-pad_x]
            ipx, ipy = px + pad_x, py + pad_y
            ipw, iph = inner.shape[1], inner.shape[0]

            if is_thermal:
                all_defects.extend(self._detect_thermal_hotspots(inner, ipx, ipy, ipw, iph, panel_id))
                diode = self._detect_bypass_diode(inner, ipx, ipy, ipw, iph, panel_id, is_thermal=True)
                if diode:
                    all_defects.append(diode)
            else:
                all_defects.extend(self._detect_optical_hotspots(inner, ipx, ipy, ipw, iph, panel_id))
                diode = self._detect_bypass_diode(inner, ipx, ipy, ipw, iph, panel_id, is_thermal=False)
                if diode:
                    all_defects.append(diode)
                all_defects.extend(self._detect_cracks(inner, ipx, ipy, ipw, iph, panel_id))
                all_defects.extend(self._detect_soiling(inner, ipx, ipy, ipw, iph, panel_id))

        return all_defects

    def _detect_thermal_hotspots(self, roi: np.ndarray, px: int, py: int, pw: int, ph: int, panel_id: str) -> List[Dict[str, Any]]:
        """Detects localized overheating with Delta-T radiometric calculation."""
        defects = []
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        r_chan = roi[:, :, 2] if len(roi.shape) == 3 else gray
        heat = np.maximum(gray, r_chan)

        mean_t, max_t = float(np.mean(heat)), float(np.max(heat))

        if (max_t - mean_t) >= 28.0 and max_t >= 190.0:
            th = int(mean_t + (max_t - mean_t) * 0.50)
            _, mask = cv2.threshold(heat, th, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours:
                area = cv2.contourArea(c)
                if 25 <= area <= (pw * ph * 0.25):
                    bx, by, bw, bh = cv2.boundingRect(c)
                    delta_t = round((float(np.max(heat[by:by+bh, bx:bx+bw])) - mean_t) * 0.25, 1)
                    sev = "CRITICAL" if delta_t >= 10.0 else "WARNING"

                    defects.append({
                        "id": f"DEF-HS_{panel_id}",
                        "panel_id": panel_id,
                        "type": "HOTSPOT",
                        "type_name": DEFECT_CLASSES["HOTSPOT"]["name"],
                        "severity": sev,
                        "confidence": round(min(0.98, 0.78 + (delta_t / 35.0)), 2),
                        "bbox_global": (px + bx, py + by, bw, bh),
                        "bbox_local": (bx, by, bw, bh),
                        "area_px": int(area),
                        "delta_t_celsius": delta_t,
                        "est_power_loss_pct": DEFECT_CLASSES["HOTSPOT"]["avg_power_loss_pct"],
                        "description": f"Thermal hotspot detected with Delta-T of +{delta_t} deg C above baseline."
                    })
                    break
        return defects

    def _detect_optical_hotspots(self, roi: np.ndarray, px: int, py: int, pw: int, ph: int, panel_id: str) -> List[Dict[str, Any]]:
        """Detects visible burn shunts or deep discoloration spots in RGB optical mode."""
        defects = []
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        _, th = cv2.threshold(v, 245, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            if 35 <= area <= (pw * ph * 0.15):
                bx, by, bw, bh = cv2.boundingRect(c)
                aspect = float(bw) / float(bh) if bh > 0 else 0
                if 0.40 <= aspect <= 2.5 and min(bw, bh) >= 8:
                    defects.append({
                        "id": f"DEF-HS_{panel_id}",
                        "panel_id": panel_id,
                        "type": "HOTSPOT",
                        "type_name": DEFECT_CLASSES["HOTSPOT"]["name"],
                        "severity": "WARNING",
                        "confidence": 0.85,
                        "bbox_global": (px + bx, py + by, bw, bh),
                        "bbox_local": (bx, by, bw, bh),
                        "area_px": int(area),
                        "delta_t_celsius": 6.5,
                        "est_power_loss_pct": DEFECT_CLASSES["HOTSPOT"]["avg_power_loss_pct"],
                        "description": "Visual surface burn/hotspot discoloration on silicon cell."
                    })
                    break
        return defects

    def _detect_bypass_diode(self, roi: np.ndarray, px: int, py: int, pw: int, ph: int, panel_id: str, is_thermal: bool) -> Optional[Dict[str, Any]]:
        """Identifies bypass diode failure causing an entire 1/3 vertical sub-string to overheat."""
        if pw < 40 or ph < 40:
            return None

        heat = np.maximum(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), roi[:, :, 2]) if (is_thermal and len(roi.shape) == 3) else cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        third_w = pw // 3
        means = [
            float(np.mean(heat[:, 0:third_w])),
            float(np.mean(heat[:, third_w:2*third_w])),
            float(np.mean(heat[:, 2*third_w:pw]))
        ]
        overall = sum(means) / 3.0
        max_diff = max(abs(m - overall) for m in means)

        if max_diff >= 18.0:
            defective_idx = int(np.argmax([abs(m - overall) for m in means]))
            sub_x, sub_w = defective_idx * third_w, third_w
            delta_val = round(max_diff * 0.28, 1)

            return {
                "id": f"DEF-DIODE_{panel_id}",
                "panel_id": panel_id,
                "type": "BYPASS_DIODE_FAILURE",
                "type_name": DEFECT_CLASSES["BYPASS_DIODE_FAILURE"]["name"],
                "severity": "CRITICAL",
                "confidence": 0.94,
                "bbox_global": (px + sub_x, py, sub_w, ph),
                "bbox_local": (sub_x, 0, sub_w, ph),
                "area_px": sub_w * ph,
                "delta_t_celsius": delta_val,
                "est_power_loss_pct": DEFECT_CLASSES["BYPASS_DIODE_FAILURE"]["avg_power_loss_pct"],
                "description": f"Sub-string #{defective_idx+1} bypassed/inactive due to diode short-circuit. Loss of 1/3 module capacity."
            }
        return None

    def _detect_cracks(self, roi: np.ndarray, px: int, py: int, pw: int, ph: int, panel_id: str) -> List[Dict[str, Any]]:
        """Detects micro-cracks and fracture lines across the wafer surface."""
        defects = []
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        edges = cv2.Canny(gray, 70, 180)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=35, minLineLength=25, maxLineGap=4)

        if lines is not None:
            diagonal_lines = []
            for line in lines:
                l = line.flatten()
                if len(l) < 4:
                    continue
                x1, y1, x2, y2 = int(l[0]), int(l[1]), int(l[2]), int(l[3])
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                if 25.0 < angle < 65.0:
                    diagonal_lines.append((x1, y1, x2, y2))

            if len(diagonal_lines) >= 3:
                all_x = [l[0] for l in diagonal_lines] + [l[2] for l in diagonal_lines]
                all_y = [l[1] for l in diagonal_lines] + [l[3] for l in diagonal_lines]
                min_x, max_x = max(0, min(all_x) - 4), min(pw, max(all_x) + 4)
                min_y, max_y = max(0, min(all_y) - 4), min(ph, max(all_y) + 4)

                defects.append({
                    "id": f"DEF-CRK_{panel_id}",
                    "panel_id": panel_id,
                    "type": "SURFACE_CRACK",
                    "type_name": DEFECT_CLASSES["SURFACE_CRACK"]["name"],
                    "severity": "WARNING",
                    "confidence": 0.88,
                    "bbox_global": (px + min_x, py + min_y, max_x - min_x, max_y - min_y),
                    "bbox_local": (min_x, min_y, max_x - min_x, max_y - min_y),
                    "area_px": (max_x - min_x) * (max_y - min_y),
                    "delta_t_celsius": 2.2,
                    "est_power_loss_pct": DEFECT_CLASSES["SURFACE_CRACK"]["avg_power_loss_pct"],
                    "description": "Irregular diagonal micro-fractures detected on module frontsheet."
                })
        return defects

    def _detect_soiling(self, roi: np.ndarray, px: int, py: int, pw: int, ph: int, panel_id: str) -> List[Dict[str, Any]]:
        """Detects uneven dust accumulation, pollen, and bird droppings."""
        defects = []
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (17, 17), 0)
            diff = cv2.absdiff(gray, blurred)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours:
                area = cv2.contourArea(c)
                if (pw * ph * 0.05) <= area <= (pw * ph * 0.35):
                    bx, by, bw, bh = cv2.boundingRect(c)
                    defects.append({
                        "id": f"DEF-SOIL_{panel_id}",
                        "panel_id": panel_id,
                        "type": "SOILING_DUST",
                        "type_name": DEFECT_CLASSES["SOILING_DUST"]["name"],
                        "severity": "ADVISORY",
                        "confidence": 0.80,
                        "bbox_global": (px + bx, py + by, bw, bh),
                        "bbox_local": (bx, by, bw, bh),
                        "area_px": int(area),
                        "delta_t_celsius": 1.5,
                        "est_power_loss_pct": DEFECT_CLASSES["SOILING_DUST"]["avg_power_loss_pct"],
                        "description": "Localized soiling / dust deposition obstructing solar irradiance."
                    })
                    break
        return defects
