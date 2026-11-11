# -*- coding: utf-8 -*-
"""
CloudPath AI - Final B.Tech Project Report Generator
Produces:
  1. CloudPath_AI_Final_Project_Report.docx
  2. CloudPath_AI_Final_Project_Report.pdf
"""

import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

BASE_DIR = r"D:\bengaluru.csv"
DIAG_DIR = os.path.join(BASE_DIR, "CloudPath_AI_Diagrams")
DOCX_PATH = os.path.join(BASE_DIR, "CloudPath_AI_Final_Project_Report.docx")
PDF_PATH = os.path.join(BASE_DIR, "CloudPath_AI_Final_Project_Report.pdf")

# ------------------------------------------------------------------------------
# REPORTLAB NUMBERED CANVAS CLASS
# ------------------------------------------------------------------------------
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        if self._pageNumber == 1:
            return  # Suppress header and footer on Title Page

        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))

        # Running Header (Top)
        self.drawString(54, 750, "CloudPath AI: AI-Based Cloud Movement Tracking and Rainfall Prediction System")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 744, 558, 744)

        # Running Footer (Bottom)
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 40, page_str)
        self.drawString(54, 40, "Department of Computer Science & Engineering - B.Tech Project Report")
        self.line(54, 50, 558, 50)

        self.restoreState()


# ------------------------------------------------------------------------------
# DOCX HELPER FUNCTIONS
# ------------------------------------------------------------------------------
def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def add_styled_paragraph(doc, text, style='Normal', space_after=6, line_spacing=1.15, align=WD_ALIGN_PARAGRAPH.JUSTIFY, bold=False, italic=False, size_pt=12, color_rgb=(0,0,0)):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = line_spacing
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(size_pt)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor(*color_rgb)
    return p

def add_heading_1(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(18)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(16)
    run.bold = True
    run.font.color.rgb = RGBColor(15, 23, 42) # Slate 900
    return p

def add_heading_2(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(13)
    run.bold = True
    run.font.color.rgb = RGBColor(30, 41, 59)
    return p

def add_heading_3(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    run.bold = True
    run.font.color.rgb = RGBColor(51, 65, 85)
    return p

def add_figure_to_docx(doc, img_filename, caption_text, explanation_text):
    img_path = os.path.join(DIAG_DIR, img_filename)
    if os.path.exists(img_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(12)
        p_img.paragraph_format.space_after = Pt(4)
        run_img = p_img.add_run()
        run_img.add_picture(img_path, width=Inches(5.8))
        
        # Caption
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_after = Pt(8)
        p_cap.paragraph_format.keep_with_next = True
        run_cap = p_cap.add_run(caption_text)
        run_cap.font.name = 'Times New Roman'
        run_cap.font.size = Pt(10.5)
        run_cap.bold = True
        run_cap.font.color.rgb = RGBColor(30, 41, 59)
        
        # Detailed narrative explanation
        p_exp = doc.add_paragraph()
        p_exp.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p_exp.paragraph_format.space_after = Pt(12)
        p_exp.paragraph_format.line_spacing = 1.15
        run_exp = p_exp.add_run(explanation_text)
        run_exp.font.name = 'Times New Roman'
        run_exp.font.size = Pt(11)
        run_exp.font.color.rgb = RGBColor(51, 65, 85)
    else:
        print(f"Warning: Image file not found: {img_path}")

def add_table_to_docx(doc, headers, data, col_widths=None):
    table = doc.add_table(rows=len(data) + 1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    # Header Row
    hdr_cells = table.rows[0].cells
    for i, header_text in enumerate(headers):
        hdr_cells[i].text = header_text
        set_cell_background(hdr_cells[i], "1E293B") # Dark slate
        set_cell_margins(hdr_cells[i], top=120, bottom=120, left=150, right=150)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            run.font.name = 'Times New Roman'
            run.font.size = Pt(10)
            run.bold = True
            run.font.color.rgb = RGBColor(255, 255, 255)

    # Data Rows
    for r_idx, row_data in enumerate(data):
        row_cells = table.rows[r_idx + 1].cells
        bg_color = "F8FAFC" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            row_cells[c_idx].text = str(val)
            set_cell_background(row_cells[c_idx], bg_color)
            set_cell_margins(row_cells[c_idx], top=100, bottom=100, left=150, right=150)
            p = row_cells[c_idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx > 0 else WD_ALIGN_PARAGRAPH.LEFT
            for run in p.runs:
                run.font.name = 'Times New Roman'
                run.font.size = Pt(9.5)
                run.font.color.rgb = RGBColor(30, 41, 59)

    # Set Widths if specified
    if col_widths:
        for row in table.rows:
            for idx, width in enumerate(col_widths):
                row.cells[idx].width = Inches(width)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

print("Helper definitions loaded.")
# ------------------------------------------------------------------------------
# REPORT DATA AND TEXT DEFINITIONS
# ------------------------------------------------------------------------------

REPORT_TITLE = "CloudPath AI – AI-Based Cloud Movement Tracking and Rainfall Prediction System"
SUBTITLE = "A Project Report Submitted in Partial Fulfillment of the Requirements\nfor the Award of the Degree of"
DEGREE = "BACHELOR OF TECHNOLOGY\nIN\nCOMPUTER SCIENCE AND ENGINEERING"
REGION_INFO = "Bengaluru Urban Region, Karnataka, India (12.9716° N, 77.5946° E)"

ABSTRACT_TEXT = (
    "Precipitation forecasting and localized convective cloud monitoring represent critical challenges in modern "
    "operational meteorology, urban disaster management, and environmental monitoring. Traditional Numerical "
    "Weather Prediction (NWP) frameworks struggle to deliver high-resolution, low-latency nowcasts for rapidly "
    "evolving convective storm events over complex tropical urban microclimates such as the Bengaluru Urban Region "
    "(12.9716° N, 77.5946° E). This project presents CloudPath AI, an end-to-end, dual-pipeline meteorological "
    "intelligence framework combining high-resolution geostationary satellite image processing with machine-learning-driven "
    "quantitative precipitation estimation. The first pipeline ingests Level-2B Cloud Cover NetCDF-4 imagery from "
    "ISRO's state-of-the-art INSAT-3DS geostationary satellite (spanning the entire active month of July 2025 across "
    "1,448 observations) cropped to a 162 x 159 pixel spatial domain (~650 km x 650 km). Using Otsu adaptive thresholding, "
    "8-connectivity Connected Component Analysis (CCA), and the Hungarian (Kuhn-Munkres) global optimization algorithm, "
    "the system detects 16,321 discrete cloud formations, resolves spatial correspondences across successive 30-minute "
    "frames, and constructs 2,831 persistent motion tracks, capturing cloud kinematics, mean speeds of 70.1 km/h, and "
    "directional heading vectors. The second pipeline features a dual-stage machine learning inference engine utilizing "
    "tuned Extreme Gradient Boosting (XGBoost) models trained on 96,431 hourly surface meteorological records (2009–2020) "
    "encompassing 25 atmospheric features. The classification stage identifies rain occurrence with 94.13% accuracy, "
    "an F1-score of 0.8349, and an ROC-AUC of 0.9753, while the quantitative regression stage predicts precipitation depth "
    "with a Mean Absolute Error (MAE) of 0.0597 mm and an R² of 0.7598. An interactive, dark-themed operational web dashboard "
    "integrates both pipelines to provide simultaneous spatial tracking and real-time precipitation nowcasting, demonstrating "
    "a scalable, production-ready framework for urban meteorological decision support."
)

KEYWORDS_TEXT = "INSAT-3DS; Cloud Tracking; Hungarian Algorithm; Extreme Gradient Boosting (XGBoost); Precipitation Nowcasting; Connected Component Analysis; Satellite Remote Sensing; Urban Hydrometeorology."

print("Report text constants loaded.")
