# UAV-Based Solar Panel Visual Inspection System

An automated, computer-vision pipeline for capturing and analyzing aerial drone survey imagery of photovoltaic (PV) solar farms. The system supports multi-modal inspection (Thermal IR & RGB Optical) to detect cell fractures, bypass diode failures, hotspots, and soiling.

---

## 🏛 Project Architecture & Team Modularity

```text
UAV-Based-Solar-Panel-Visual-Inspection-System/
│
├── data/
│   ├── original/           # Raw UAV aerial frames (Thermal IR & RGB Optical)
│   ├── processed/          # CLAHE-enhanced and segmented ROI crops
│   └── ground_truth/       # Defect annotations & benchmarks
│
├── outputs/
│   ├── annotated/          # Bounding-box overlaid survey frames
│   └── reports/            # Inspection logs (CSV, HTML, PDF)
│
├── src/
│   ├── __init__.py         # Package exports
│   ├── preprocessing.py    # [Member 2] Image loading, CLAHE, perspective deskew, panel segmentation
│   ├── detection.py        # [Member 3] Multi-modal defect classification & anomaly detection
│   ├── evaluation.py       # [Member 3] IoU, Precision, Recall, and F1-score evaluation
│   ├── visualization.py    # [Member 4] Bounding-box annotations & thermal colormaps
│   ├── reporting.py        # [Member 4] Executive HTML & engineering PDF audit reports
│   └── pipeline.py         # End-to-end inspection orchestrator
│
├── tests/
│   ├── test_preprocessing.py # [Member 2] Tests for preprocessing & segmentation
│   ├── test_detection.py     # [Member 3] Tests for defect detection engine
│   └── test_reporting.py     # [Member 4] Tests for report generation & exports
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🧩 Member 2 Scope: Preprocessing & Panel Segmentation

### Key Modules Implemented
- **[`ImagePreprocessor`](src/preprocessing.py)**:
  - `load_image(image_path)`: Robust loading of drone imagery.
  - `detect_modality(image, filename)`: Auto-classification of Thermal IR vs. RGB Optical frames.
  - `apply_clahe(image, clip_limit, tile_grid_size)`: Contrast Limited Adaptive Histogram Equalization for hotspot & cell visibility.
  - `extract_edges(image)`: Multi-scale Sobel/Canny edge extraction for solar array grid lines.
  - `deskew(image)`: Perspective rectification to correct for UAV camera pitch and yaw angles.

- **[`PanelSegmenter`](src/preprocessing.py)**:
  - `segment_panels(image)`: Contour-based & adaptive grid segmentation isolating individual PV panels/cells into indexed regions of interest (`PV-Rxx-Mxx`).

### Verification & Testing
To execute the Member 2 test suite:
```bash
python -m unittest tests/test_preprocessing.py
```
*(All 4 unit tests passing: load_image, detect_modality, apply_clahe, panel_segmenter)*

---

## 🚀 Setup & Installation

```bash
# Clone the repository
git clone https://github.com/Akashat-Tiwari/UAV-Based-Solar-Panel-Visual-Inspection-System.git
cd UAV-Based-Solar-Panel-Visual-Inspection-System

# Install dependencies
pip install -r requirements.txt
```
