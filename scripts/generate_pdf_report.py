#!/usr/bin/env python3
"""Compile report.md to a formatted report.pdf using ReportLab."""
import sys
from pathlib import Path
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib import colors

ROOT_DIR = Path(__file__).parent.parent.resolve()

def build_pdf():
    md_file = ROOT_DIR / "report.md"
    pdf_file = ROOT_DIR / "report.pdf"

    if not md_file.exists():
        print(f"Error: {md_file} not found.")
        return

    doc = SimpleDocTemplate(
        str(pdf_file),
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#064e3b'),
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#4b5563'),
        spaceAfter=12,
    )
    h2_style = ParagraphStyle(
        'DocH2',
        parent=styles['Heading2'],
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#1f2937'),
        spaceBefore=10,
        spaceAfter=6,
    )
    h3_style = ParagraphStyle(
        'DocH3',
        parent=styles['Heading3'],
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#111827'),
        spaceBefore=8,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#1f2937'),
        spaceAfter=6,
    )
    bullet_style = ParagraphStyle(
        'DocBullet',
        parent=body_style,
        leftIndent=15,
        bulletIndent=5,
        spaceAfter=3,
    )
    table_cell_style = ParagraphStyle(
        'DocTableCell',
        parent=styles['Normal'],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#111827'),
    )
    table_header_style = ParagraphStyle(
        'DocTableHeader',
        parent=styles['Normal'],
        fontSize=8,
        leading=11,
        fontName='Helvetica-Bold',
        textColor=colors.white,
    )

    elements = []
    lines = md_file.read_text(encoding="utf-8").splitlines()

    in_table = False
    table_data = []

    def render_table(t_data):
        if not t_data or len(t_data) == 0 or len(t_data[0]) == 0:
            return None
        num_cols = len(t_data[0])
        col_w = 530.0 / num_cols
        t = Table(t_data, colWidths=[col_w] * num_cols)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#064e3b')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d1d5db')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f9fafb'), colors.white]),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        return t

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_table and table_data:
                tbl = render_table(table_data)
                if tbl:
                    elements.append(tbl)
                    elements.append(Spacer(1, 8))
                in_table = False
                table_data = []
            continue

        if stripped.startswith("|") and "|" in stripped[1:]:
            if ":---" in stripped or "---:" in stripped:
                continue
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            if not cells:
                continue
            if not in_table:
                in_table = True
                table_data.append([Paragraph(c, table_header_style) for c in cells])
            else:
                table_data.append([Paragraph(c, table_cell_style) for c in cells])
            continue
        elif in_table:
            tbl = render_table(table_data)
            if tbl:
                elements.append(tbl)
                elements.append(Spacer(1, 8))
            in_table = False
            table_data = []

        if stripped.startswith("# "):
            elements.append(Paragraph(stripped[2:], title_style))
        elif stripped.startswith("## "):
            elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e5e7eb'), spaceBefore=6, spaceAfter=6))
            elements.append(Paragraph(stripped[3:], h2_style))
        elif stripped.startswith("### "):
            elements.append(Paragraph(stripped[4:], h3_style))
        elif stripped.startswith("- ") or stripped.startswith("* "):
            elements.append(Paragraph(f"&bull; {stripped[2:]}", bullet_style))
        elif stripped.startswith("1. ") or stripped.startswith("2. ") or stripped.startswith("3. ") or stripped.startswith("4. "):
            elements.append(Paragraph(stripped, bullet_style))
        elif stripped.startswith("**") and stripped.endswith("**"):
            elements.append(Paragraph(stripped, subtitle_style))
        elif stripped.startswith("```"):
            continue
        else:
            import re, html
            # Escape HTML characters first
            escaped = html.escape(stripped)
            # Replace **bold** with <b>bold</b>
            formatted = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', escaped)
            formatted = re.sub(r'\*(.+?)\*', r'<i>\1</i>', formatted)
            formatted = re.sub(r'`(.+?)`', r'<font face="Courier">\1</font>', formatted)
            try:
                elements.append(Paragraph(formatted, body_style))
            except Exception:
                elements.append(Paragraph(html.escape(stripped), body_style))

    doc.build(elements)
    print(f"✅ Generated {pdf_file.name} successfully.")

if __name__ == "__main__":
    build_pdf()
