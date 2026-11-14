"""
Visualization and Inspection Annotation Module.
Draws color-coded bounding boxes, severity badges, and comparative diagnostic figures.
"""

import os
from typing import List, Dict, Any
import cv2
import numpy as np

from .detection import DEFECT_CLASSES


class Visualizer:
    """Renders visual detection overlays, bounding boxes, and diagnostic images."""

    @staticmethod
    def draw_annotations(
        img_bgr: np.ndarray,
        panels: List[Dict[str, Any]],
        defects: List[Dict[str, Any]],
        show_panel_ids: bool = True
    ) -> np.ndarray:
        """
        Draws panel boundaries and color-coded defect annotations onto the imagery.
        """
        annotated = img_bgr.copy()

        # 1. Draw individual solar panel grid boundaries
        for p in panels:
            px, py, pw, ph = p["bbox"]
            cv2.rectangle(annotated, (px, py), (px + pw, py + ph), (220, 180, 0), 1, cv2.LINE_AA)
            if show_panel_ids:
                cv2.putText(
                    annotated, p["id"], (px + 4, py + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (240, 240, 240), 1, cv2.LINE_AA
                )

        # 2. Draw defects with thick borders and contrast tags
        for d in defects:
            gx, gy, gw, gh = d["bbox_global"]
            meta = DEFECT_CLASSES.get(d["type"], {"color_bgr": (0, 0, 255), "name": d["type"]})
            color = meta["color_bgr"]

            # Main bounding box
            cv2.rectangle(annotated, (gx, gy), (gx + gw, gy + gh), color, 2, cv2.LINE_AA)

            # Corner accents
            c_len = min(12, gw // 3, gh // 3)
            cv2.line(annotated, (gx, gy), (gx + c_len, gy), color, 3)
            cv2.line(annotated, (gx, gy), (gx, gy + c_len), color, 3)
            cv2.line(annotated, (gx + gw, gy), (gx + gw - c_len, gy), color, 3)
            cv2.line(annotated, (gx + gw, gy), (gx + gw, gy + c_len), color, 3)

            # Label text
            label = f"{d['severity'][:4]}: {meta['name'].split('/')[0].strip()} ({d['confidence']*100:.0f}%)"
            if "delta_t_celsius" in d and d["delta_t_celsius"] > 0:
                label += f" +{d['delta_t_celsius']}C"

            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            y_top = max(0, gy - lh - 6)
            cv2.rectangle(annotated, (gx, y_top), (gx + lw + 6, gy), (20, 20, 20), -1)
            cv2.rectangle(annotated, (gx, y_top), (gx + lw + 6, gy), color, 1)
            cv2.putText(
                annotated, label, (gx + 3, gy - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA
            )

        return annotated

    @staticmethod
    def create_side_by_side(raw_bgr: np.ndarray, processed_bgr: np.ndarray, annotated_bgr: np.ndarray) -> np.ndarray:
        """Combines raw, preprocessed, and annotated frames into a 3-panel comparative view."""
        h, w = raw_bgr.shape[:2]
        target_w = 480
        target_h = int(h * (target_w / w))

        r1 = cv2.resize(raw_bgr, (target_w, target_h))
        r2 = cv2.resize(processed_bgr, (target_w, target_h))
        r3 = cv2.resize(annotated_bgr, (target_w, target_h))

        # Add title headers
        cv2.putText(r1, "1. Raw Drone Image", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(r2, "2. CLAHE Preprocessed", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(r3, "3. AI Anomaly Overlay", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return np.hstack([r1, r2, r3])

    @staticmethod
    def save_image(img_bgr: np.ndarray, path: str):
        """Saves image to disk creating directories if necessary."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        cv2.imwrite(path, img_bgr)
