"""
Evaluation and Benchmarking Module.
Compares automated AI detections against ground-truth annotations to compute
Precision, Recall, F1-Score, and Intersection-over-Union (IoU) metrics.
"""

import os
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np


class Evaluator:
    """Evaluates detection model performance against annotated ground truth."""

    @staticmethod
    def load_ground_truth(csv_path: str) -> List[Dict[str, Any]]:
        """Loads ground truth bounding boxes and defect classes from annotations.csv."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Ground truth file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        records = []
        for _, row in df.iterrows():
            records.append({
                "image_name": str(row["image_name"]),
                "panel_id": str(row["panel_id"]),
                "defect_type": str(row["defect_type"]),
                "severity": str(row.get("severity", "WARNING")),
                "bbox": (int(row["bbox_x"]), int(row["bbox_y"]), int(row["bbox_w"]), int(row["bbox_h"]))
            })
        return records

    @staticmethod
    def compute_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
        """Computes Intersection-over-Union between two bounding boxes (x, y, w, h)."""
        ax, ay, aw, ah = boxA
        bx, by, bw, bh = boxB

        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)

        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter_area = inter_w * inter_h

        areaA = aw * ah
        areaB = bw * bh
        union_area = float(areaA + areaB - inter_area)

        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    @classmethod
    def evaluate(cls, detections: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]], iou_threshold: float = 0.25) -> Dict[str, Any]:
        """
        Evaluates detections against ground truth annotations.
        Matches defects by class and spatial IoU overlap.
        """
        tp = 0
        fp = 0
        matched_gt = set()

        for det in detections:
            det_img = det.get("image_name", "")
            det_type = det["type"]
            det_box = det["bbox_global"]

            best_iou = 0.0
            best_gt_idx = -1

            for idx, gt in enumerate(ground_truth):
                if idx in matched_gt:
                    continue
                # Match image if provided
                if det_img and gt["image_name"] and det_img != gt["image_name"]:
                    continue
                # Match defect class
                if det_type != gt["defect_type"]:
                    continue

                iou = cls.compute_iou(det_box, gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou >= iou_threshold and best_gt_idx != -1:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1

        fn = len(ground_truth) - len(matched_gt)

        precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
        f1_score = round(2 * (precision * recall) / (precision + recall), 4) if (precision + recall) > 0 else 0.0

        return {
            "total_ground_truth": len(ground_truth),
            "total_detections": len(detections),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "iou_threshold": iou_threshold
        }
