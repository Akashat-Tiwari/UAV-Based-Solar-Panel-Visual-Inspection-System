# Member 4: Visualization & Inspection Reporting

## Assigned Module Scope
- **Source Code**: `src/visualization.py` and `src/reporting.py`
- **Unit Tests**: `tests/test_reporting.py`
- **Outputs**: `outputs/reports/`

## Key Responsibilities
1. **`Visualizer` (`src/visualization.py`)**:
   - Draw annotated bounding boxes, defect tags, and severity badges on UAV frames.
   - Generate side-by-side thermal/optical comparison plots.
   - Save annotated frames to `outputs/annotated/`.

2. **`ReportGenerator` (`src/reporting.py`)**:
   - Aggregate fleet metrics: Total Modules, Fleet Health Index (%), Defective Count, Power Loss (kW), Financial Loss ($).
   - Export CSV inspection logs to `outputs/reports/inspection_results.csv`.
   - Generate executive HTML audit reports with responsive tables and styling.
   - Generate engineering PDF audit reports using ReportLab.

## How to Test
Run:
```bash
python -m unittest tests/test_reporting.py
```

## GitHub Collaboration Workflow
```bash
git checkout -b member4-reporting
git add src/visualization.py src/reporting.py tests/test_reporting.py
git commit -m "Member 4: update PDF styling and executive summaries"
git push origin member4-reporting
```
