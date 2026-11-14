"""
Reporting and Analytics Module.
Exports engineering inspection reports (PDF & HTML) and CSV maintenance work orders.
"""

import os
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from .detection import SEVERITY_LEVELS


class ReportGenerator:
    """Compiles inspection findings into professional engineering audit reports."""

    @staticmethod
    def calculate_summary(panels: List[Dict[str, Any]], defects: List[Dict[str, Any]], metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Calculates quantitative performance, health index, and capacity degradation."""
        total_panels = len(panels)
        defective_panel_ids = set(d["panel_id"] for d in defects)
        num_defective = len(defective_panel_ids)
        healthy = max(0, total_panels - num_defective)
        health_pct = round((healthy / total_panels * 100.0) if total_panels else 100.0, 1)

        crit_count = sum(1 for d in defects if d["severity"] == "CRITICAL")
        warn_count = sum(1 for d in defects if d["severity"] == "WARNING")
        adv_count = sum(1 for d in defects if d["severity"] == "ADVISORY")

        # Nominal 400W module rating
        power_loss_kw = round(sum(d.get("est_power_loss_pct", 10.0) * 0.40 for d in defects) / 10.0, 2)
        annual_financial_loss = round(power_loss_kw * 5.2 * 365 * 0.08, 2)

        return {
            "metadata": metadata or {"drone_model": "DJI Matrice 300 RTK / H20T", "altitude_m": 35.0, "latitude": 28.6139, "longitude": 77.2090},
            "total_panels": total_panels,
            "healthy_panels": healthy,
            "defective_panels": num_defective,
            "health_index_pct": health_pct,
            "total_defects": len(defects),
            "critical_defects": crit_count,
            "warning_defects": warn_count,
            "advisory_defects": adv_count,
            "total_power_loss_kw": power_loss_kw,
            "annual_financial_loss_usd": annual_financial_loss,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    @staticmethod
    def export_csv(output_csv_path: str, defects: List[Dict[str, Any]], image_name: str = "") -> str:
        """Exports anomaly registry to CSV (matches inspection_results.csv architecture)."""
        os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
        rows = []
        for d in defects:
            gx, gy, gw, gh = d.get("bbox_global", (0, 0, 0, 0))
            rows.append({
                "defect_id": d["id"],
                "image_name": d.get("image_name", image_name),
                "panel_id": d["panel_id"],
                "defect_type": d["type"],
                "defect_name": d["type_name"],
                "severity": d["severity"],
                "confidence": d.get("confidence", 0.9),
                "delta_t_celsius": d.get("delta_t_celsius", 0.0),
                "est_power_loss_pct": d.get("est_power_loss_pct", 0.0),
                "bbox_x": gx,
                "bbox_y": gy,
                "bbox_w": gw,
                "bbox_h": gh,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

        df = pd.DataFrame(rows)
        df.to_csv(output_csv_path, index=False)
        return output_csv_path

    @staticmethod
    def export_html(output_html_path: str, summary: Dict[str, Any], defects: List[Dict[str, Any]]) -> str:
        """Generates a standalone responsive HTML audit report."""
        os.makedirs(os.path.dirname(os.path.abspath(output_html_path)), exist_ok=True)
        meta = summary.get("metadata", {})

        rows_html = ""
        for d in defects:
            dt = f"+{d.get('delta_t_celsius', '-')} deg C" if 'delta_t_celsius' in d else "-"
            action = SEVERITY_LEVELS.get(d["severity"], {}).get("action", "Inspect module")
            badge_class = "crit" if d["severity"] == "CRITICAL" else "warn" if d["severity"] == "WARNING" else "adv"
            rows_html += f"""
            <tr>
                <td><code>{d['id']}</code></td>
                <td><strong>{d['panel_id']}</strong></td>
                <td>{d['type_name']}</td>
                <td><span class="badge {badge_class}">{d['severity']}</span></td>
                <td>{d.get('confidence', 0.9)*100:.0f}%</td>
                <td>{dt}</td>
                <td>{action}</td>
            </tr>
            """

        if not defects:
            rows_html = "<tr><td colspan='7' style='text-align:center; padding:18px; color:#10b981;'>All photovoltaic modules operating at peak efficiency.</td></tr>"

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Solar UAV Inspection Audit Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; padding: 30px; margin: 0; }}
        .container {{ max-width: 1050px; margin: 0 auto; }}
        header {{ border-bottom: 2px solid #3b82f6; padding-bottom: 12px; margin-bottom: 20px; }}
        .kpis {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 24px; }}
        .kpi {{ background: #1e293b; padding: 16px; border-radius: 8px; border: 1px solid #334155; text-align: center; }}
        .kpi h2 {{ margin: 6px 0 0 0; font-size: 24px; }}
        .card {{ background: #1e293b; padding: 20px; border-radius: 8px; border: 1px solid #334155; margin-bottom: 20px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ background: #0f172a; color: #cbd5e1; }}
        .badge {{ padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 11px; }}
        .crit {{ background: #dc2626; color: white; }}
        .warn {{ background: #d97706; color: white; }}
        .adv {{ background: #2563eb; color: white; }}
        .btn {{ background: #3b82f6; color: white; border: none; padding: 10px 18px; border-radius: 6px; cursor: pointer; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1 style="margin: 0 0 8px 0;">Solar UAV Inspection Audit Report</h1>
            <div style="color: #94a3b8; font-size: 13px;">Target: {meta.get('drone_model', 'UAV Platform')} | GPS: {meta.get('latitude', 28.61)} N, {meta.get('longitude', 77.20)} E | Audit Date: {summary['generated_at']}</div>
        </header>

        <div class="kpis">
            <div class="kpi"><span style="color:#94a3b8; font-size:12px;">TOTAL MODULES</span><h2 style="color:#60a5fa;">{summary['total_panels']}</h2></div>
            <div class="kpi"><span style="color:#94a3b8; font-size:12px;">HEALTH INDEX</span><h2 style="color:#34d399;">{summary['health_index_pct']}%</h2></div>
            <div class="kpi"><span style="color:#94a3b8; font-size:12px;">CRITICAL DEFICIENCIES</span><h2 style="color:#f87171;">{summary['critical_defects']}</h2></div>
            <div class="kpi"><span style="color:#94a3b8; font-size:12px;">EST. CAPACITY LOSS</span><h2 style="color:#fbbf24;">{summary['total_power_loss_kw']} kW</h2></div>
        </div>

        <div class="card">
            <h3 style="margin-top:0;">Photovoltaic Defect Registry & Corrective Action Log</h3>
            <table>
                <thead>
                    <tr><th>Defect ID</th><th>Module</th><th>Classification</th><th>Severity</th><th>Confidence</th><th>Delta-T</th><th>Recommended Action</th></tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>

        <div style="text-align: right; margin-top: 15px;">
            <button class="btn" onclick="window.print()">Print / Save as PDF</button>
        </div>
    </div>
</body>
</html>"""

        with open(output_html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return output_html_path

    @staticmethod
    def export_pdf(output_pdf_path: str, summary: Dict[str, Any], defects: List[Dict[str, Any]]) -> str:
        """Generates formal multi-page engineering inspection PDF document."""
        os.makedirs(os.path.dirname(os.path.abspath(output_pdf_path)), exist_ok=True)
        doc = SimpleDocTemplate(output_pdf_path, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle('TStyle', parent=styles['Heading1'], fontSize=20, leading=24, textColor=colors.HexColor('#0F172A'))
        meta_style = ParagraphStyle('MStyle', parent=styles['Normal'], fontSize=10, leading=14, textColor=colors.HexColor('#64748B'))
        head_style = ParagraphStyle('HStyle', parent=styles['Heading2'], fontSize=12, leading=16, textColor=colors.HexColor('#1E293B'))
        cell_style = ParagraphStyle('CStyle', parent=styles['Normal'], fontSize=8, leading=11, textColor=colors.HexColor('#334155'))

        elements = []
        elements.append(Paragraph("AERIAL UAV SOLAR INSPECTION REPORT", title_style))
        elements.append(Paragraph(f"Autonomous Drone Audit | IEC 62446-3 Compliance | Date: {summary['generated_at']}", meta_style))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#3B82F6'), spaceAfter=14))

        # KPI Table
        kpi_data = [
            [Paragraph("<b>Modules Surveyed</b>", cell_style), Paragraph("<b>Health Index</b>", cell_style), Paragraph("<b>Critical Anomalies</b>", cell_style), Paragraph("<b>Capacity Loss</b>", cell_style)],
            [Paragraph(f"<font size=14 color='#0F172A'><b>{summary['total_panels']}</b></font>", cell_style),
             Paragraph(f"<font size=14 color='#10B981'><b>{summary['health_index_pct']}%</b></font>", cell_style),
             Paragraph(f"<font size=14 color='#EF4444'><b>{summary['critical_defects']}</b></font>", cell_style),
             Paragraph(f"<font size=14 color='#F59E0B'><b>{summary['total_power_loss_kw']} kW</b></font>", cell_style)]
        ]
        t_kpi = Table(kpi_data, colWidths=[130, 130, 130, 130])
        t_kpi.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#CBD5E1')),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(t_kpi)
        elements.append(Spacer(1, 14))

        # Defect Table
        elements.append(Paragraph("Module Defect & Corrective Action Log", head_style))
        headers = [Paragraph("<b>ID</b>", cell_style), Paragraph("<b>Module</b>", cell_style), Paragraph("<b>Defect Class</b>", cell_style), Paragraph("<b>Severity</b>", cell_style), Paragraph("<b>Delta-T</b>", cell_style), Paragraph("<b>Corrective Action</b>", cell_style)]
        tbl_data = [headers]

        if not defects:
            tbl_data.append([Paragraph("NOMINAL", cell_style), Paragraph("ALL", cell_style), Paragraph("No defects detected", cell_style), Paragraph("OPTIMAL", cell_style), Paragraph("-", cell_style), Paragraph("No action required.", cell_style)])
        else:
            for d in defects:
                action = SEVERITY_LEVELS.get(d["severity"], {}).get("action", "Inspect")
                dt = str(d.get("delta_t_celsius", "-"))
                tbl_data.append([
                    Paragraph(f"<font size=7>{d['id']}</font>", cell_style),
                    Paragraph(f"<b>{d['panel_id']}</b>", cell_style),
                    Paragraph(d['type_name'].split('/')[0], cell_style),
                    Paragraph(d['severity'], cell_style),
                    Paragraph(dt, cell_style),
                    Paragraph(f"<font size=7>{action}</font>", cell_style)
                ])

        t_defects = Table(tbl_data, colWidths=[75, 75, 110, 60, 45, 155])
        t_defects.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0F172A')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#94A3B8')),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(t_defects)
        doc.build(elements)
        return output_pdf_path
