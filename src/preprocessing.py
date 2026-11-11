"""
Image Preprocessing and Photovoltaic Panel Segmentation Module.
Handles aerial drone image filtering, CLAHE contrast enhancement,
deskewing, and panel contour localization.
"""

import os
from typing import Tuple, List, Dict, Any, Optional
import cv2
import numpy as np


class ImagePreprocessor:
    """Preprocesses raw optical RGB and thermal infrared UAV drone imagery."""

    @staticmethod
    def load_image(path: str) -> np.ndarray:
        """Loads an image with cross-platform and Unicode path compatibility."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found at path: {path}")
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to decode image from path: {path}")
        return img

    @staticmethod
    def detect_modality(img_bgr: np.ndarray, filename: str = "") -> str:
        """
        Determines whether the drone imagery is Thermal IR or Optical RGB.
        Uses filename heuristics and chromatic variance analysis.
        """
        fn_lower = filename.lower()
        if any(tag in fn_lower for tag in ["thermal", "flir", "ir_", "_ir", "radiometric", "inferno"]):
            return "THERMAL_IR"

        b, g, r = cv2.split(img_bgr)
        diff_rg = float(np.mean(np.abs(r.astype(float) - g.astype(float))))
        diff_gb = float(np.mean(np.abs(g.astype(float) - b.astype(float))))

        # Very low color difference indicates grayscale thermal/radiometric sensor
        if diff_rg < 8.0 and diff_gb < 8.0:
            return "THERMAL_GRAYSCALE"

        # Strong red/yellow dominance typical of false-color ironbow/inferno
        if np.mean(r) > np.mean(b) * 1.15 and diff_rg > 20:
            return "THERMAL_IR"

        return "RGB_OPTICAL"

    @staticmethod
    def apply_clahe(img_bgr: np.ndarray, clip_limit: float = 3.0, tile_grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Applies Contrast Limited Adaptive Histogram Equalization in LAB space.
        Reveals subtle thermal differentials and micro-cracks under harsh sunlight.
        """
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def convert_to_thermal_colormap(gray_or_bgr: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
        """Transforms grayscale radiometric intensity into false-color thermal heatmap."""
        if len(gray_or_bgr.shape) == 3:
            gray = cv2.cvtColor(gray_or_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = gray_or_bgr
        norm_gray = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return cv2.applyColorMap(norm_gray, colormap)

    @staticmethod
    def extract_edges(img_bgr: np.ndarray, low_th: int = 50, high_th: int = 150) -> np.ndarray:
        """Extracts structural wafer and aluminum frame edge boundaries."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.Canny(blurred, low_th, high_th)

    @staticmethod
    def deskew(img_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        """Aligns oblique aerial drone perspective along dominant solar string orientation."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=80, maxLineGap=10)

        if lines is None:
            return img_bgr, 0.0

        angles = []
        for line in lines:
            l = line.flatten()
            if len(l) < 4:
                continue
            x1, y1, x2, y2 = int(l[0]), int(l[1]), int(l[2]), int(l[3])
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if angle < -45:
                angle += 90
            elif angle > 45:
                angle -= 90
            angles.append(angle)

        if not angles:
            return img_bgr, 0.0

        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.5:
            return img_bgr, 0.0

        h, w = img_bgr.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated, median_angle


class PanelSegmenter:
    """Isolates individual solar panel modules from aerial drone orthophotos."""

    def __init__(self, min_area_px: int = 2500, min_aspect: float = 0.25, max_aspect: float = 4.0):
        self.min_area_px = min_area_px
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect

    def segment_panels(self, img_bgr: np.ndarray) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Segments photovoltaic modules from mounting frames and terrain.
        Returns a list of panel dictionaries and a binary mask.
        """
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        panels = []
        panel_mask = np.zeros((h, w), dtype=np.uint8)

        sorted_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if self.min_area_px <= area < (h * w * 0.9):
                x, y, bw, bh = cv2.boundingRect(c)
                sorted_contours.append((y, x, c, area))

        # Spatial sorting by rows then columns
        sorted_contours.sort(key=lambda item: (item[0] // 80, item[1]))

        for idx, (_, _, c, area) in enumerate(sorted_contours, start=1):
            x, y, bw, bh = cv2.boundingRect(c)
            aspect = float(bw) / float(bh) if bh > 0 else 0
            if self.min_aspect <= aspect <= self.max_aspect:
                roi = img_bgr[y:y+bh, x:x+bw].copy()
                cv2.drawContours(panel_mask, [c], -1, 255, -1)
                row_idx = (y // 100) + 1
                col_idx = (x // 100) + 1
                panels.append({
                    "id": f"PV-R{row_idx:02d}-M{col_idx:02d}",
                    "index": idx,
                    "bbox": (x, y, bw, bh),
                    "area_px": int(area),
                    "roi": roi,
                    "center": (x + bw // 2, y + bh // 2)
                })

        # Fallback to geometric grid if aerial perspective has low edge definition
        if len(panels) < 4:
            panels = self._fallback_grid(img_bgr)
            for p in panels:
                x, y, bw, bh = p["bbox"]
                cv2.rectangle(panel_mask, (x, y), (x + bw, y + bh), 255, -1)

        return panels, panel_mask

    def _fallback_grid(self, img_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Fallback grid segmenter for aligned solar farm strings."""
        h, w = img_bgr.shape[:2]
        rows, cols = 3, 4
        my, mx = int(h * 0.08), int(w * 0.08)
        cell_h = (h - 2 * my) // rows
        cell_w = (w - 2 * mx) // cols
        panels = []
        idx = 1
        for r in range(rows):
            for c in range(cols):
                x = mx + c * cell_w + int(cell_w * 0.04)
                y = my + r * cell_h + int(cell_h * 0.04)
                bw = int(cell_w * 0.92)
                bh = int(cell_h * 0.92)
                panels.append({
                    "id": f"PV-R{r+1:02d}-M{c+1:02d}",
                    "index": idx,
                    "bbox": (x, y, bw, bh),
                    "area_px": bw * bh,
                    "roi": img_bgr[y:y+bh, x:x+bw].copy(),
                    "center": (x + bw // 2, y + bh // 2)
                })
                idx += 1
        return panels
