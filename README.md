# Member 3: Defect Anomaly Detection & Model Evaluation

## Assigned Module Scope
- **Source Code**: `src/detection.py` and `src/evaluation.py`
- **Unit Tests**: `tests/test_detection.py`
- **Ground Truth**: `data/ground_truth/annotations.csv`

## Key Responsibilities
1. **`AnomalyDetector` (`src/detection.py`)**:
   - Detect thermal hotspots (localized overheating: Delta T >= 10 deg C).
   - Detect bypass diode failures (string-level overheating).
   - Detect optical micro-cracks, fractures, and surface soiling/dust.
   - Output defect severity (`CRITICAL`, `WARNING`, `ADVISORY`) and estimated power loss.

2. **`Evaluator` (`src/evaluation.py`)**:
   - Load ground truth annotations.
   - Calculate Bounding Box IoU (Intersection-over-Union).
   - Compute Precision, Recall, and F1-Score metrics across test surveys.

## How to Test
Run:
```bash
python -m unittest tests/test_detection.py
```

## GitHub Collaboration Workflow
```bash
git checkout -b member3-detection
git add src/detection.py src/evaluation.py tests/test_detection.py
git commit -m "Member 3: enhance hotspot sensitivity and F1 evaluation"
git push origin member3-detection
```
