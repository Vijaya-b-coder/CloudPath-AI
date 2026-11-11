# -*- coding: utf-8 -*-
"""
CloudPath AI - Project Synopsis Generator
Generates: CloudPath_AI_Project_Synopsis.docx
Following the format, style, and structure of JurisFlow_Synopsis_Final.pdf
"""

import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

OUT_PATH = r"D:\bengaluru.csv\CloudPath_AI_Project_Synopsis.docx"
DIAG_DIR = r"D:\bengaluru.csv\CloudPath_AI_Diagrams"

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

def add_body_p(doc, text, bold_prefix="", space_after=6, line_spacing=1.5):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = line_spacing
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = 'Times New Roman'
        r_pre.font.size = Pt(12)
        r_pre.bold = True
        r_pre.font.color.rgb = RGBColor(15, 23, 42)
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    run.font.color.rgb = RGBColor(30, 41, 59)
    return p

def add_main_heading(doc, text, space_before=14, space_after=6):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(14)
    run.bold = True
    run.font.color.rgb = RGBColor(15, 23, 42)
    return p

def add_sub_heading(doc, text, space_before=10, space_after=4):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    run.bold = True
    run.font.color.rgb = RGBColor(30, 41, 59)
    return p

def add_diagram(doc, img_filename, caption_text, width_in=5.6):
    img_path = os.path.join(DIAG_DIR, img_filename)
    if os.path.exists(img_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(8)
        p_img.paragraph_format.space_after = Pt(4)
        run_img = p_img.add_run()
        run_img.add_picture(img_path, width=Inches(width_in))
        
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_after = Pt(8)
        p_cap.paragraph_format.keep_with_next = True
        run_cap = p_cap.add_run(caption_text)
        run_cap.font.name = 'Times New Roman'
        run_cap.font.size = Pt(10.5)
        run_cap.bold = True
        run_cap.font.color.rgb = RGBColor(30, 41, 59)
    else:
        print(f"Warning: Image file not found: {img_path}")

def format_table(table, col_widths, headers, data):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    # Header Row
    hdr_cells = table.rows[0].cells
    for i, header_text in enumerate(headers):
        hdr_cells[i].text = header_text
        set_cell_background(hdr_cells[i], "1E293B")
        set_cell_margins(hdr_cells[i], top=100, bottom=100, left=120, right=120)
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
            set_cell_margins(row_cells[c_idx], top=80, bottom=80, left=120, right=120)
            p = row_cells[c_idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx == 0 or len(str(val)) < 15 else WD_ALIGN_PARAGRAPH.LEFT
            for run in p.runs:
                run.font.name = 'Times New Roman'
                run.font.size = Pt(9.5)
                run.font.color.rgb = RGBColor(30, 41, 59)

    for row in table.rows:
        for idx, width in enumerate(col_widths):
            row.cells[idx].width = Inches(width)

def build_synopsis():
    print("Initializing Document...")
    doc = docx.Document()

    # Set Page Setup: A4, Margins: Top 1.0", Bottom 1.0", Left 1.0", Right 1.0"
    for section in doc.sections:
        section.page_width = Inches(8.27)
        section.page_height = Inches(11.69)
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
        # Add Page Number in Footer
        footer = section.footer
        p_ft = footer.paragraphs[0]
        p_ft.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        r_ft = p_ft.add_run("Page ")
        r_ft.font.name = 'Times New Roman'
        r_ft.font.size = Pt(9.5)
        r_ft.font.color.rgb = RGBColor(100, 116, 139)
        
        fldSimple = OxmlElement('w:fldSimple')
        fldSimple.set(qn('w:instr'), 'PAGE')
        p_ft._p.append(fldSimple)

    # ==============================================================================
    # COVER PAGE
    # ==============================================================================
    p_syn = doc.add_paragraph()
    p_syn.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_syn.paragraph_format.space_before = Pt(36)
    p_syn.paragraph_format.space_after = Pt(6)
    r_syn = p_syn.add_run("SYNOPSIS FOR THE PROJECT\nON")
    r_syn.font.name = 'Times New Roman'
    r_syn.font.size = Pt(13)
    r_syn.bold = True
    r_syn.font.color.rgb = RGBColor(15, 23, 42)

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(8)
    p_title.paragraph_format.space_after = Pt(28)
    r_title = p_title.add_run('"CloudPath AI - AI-Based Cloud Movement Tracking and\nRainfall Prediction System"')
    r_title.font.name = 'Times New Roman'
    r_title.font.size = Pt(16)
    r_title.bold = True
    r_title.font.color.rgb = RGBColor(15, 23, 42)

    p_dept = doc.add_paragraph()
    p_dept.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_dept.paragraph_format.space_before = Pt(12)
    p_dept.paragraph_format.space_after = Pt(4)
    r_dept = p_dept.add_run("DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING")
    r_dept.font.name = 'Times New Roman'
    r_dept.font.size = Pt(13)
    r_dept.bold = True
    r_dept.font.color.rgb = RGBColor(30, 41, 59)

    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_inst.paragraph_format.space_after = Pt(6)
    r_inst = p_inst.add_run("BENGALURU URBAN REGION, KARNATAKA, INDIA\n(12.9716° N, 77.5946° E)")
    r_inst.font.name = 'Times New Roman'
    r_inst.font.size = Pt(11)
    r_inst.font.color.rgb = RGBColor(71, 85, 105)

    p_year = doc.add_paragraph()
    p_year.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_year.paragraph_format.space_before = Pt(6)
    p_year.paragraph_format.space_after = Pt(20)
    r_year = p_year.add_run("ACADEMIC YEAR 2026–2027\nGroup / Project Batch No. : CP-08")
    r_year.font.name = 'Times New Roman'
    r_year.font.size = Pt(12)
    r_year.bold = True
    r_year.font.color.rgb = RGBColor(15, 23, 42)

    # Student Details Table
    stud_table = doc.add_table(rows=5, cols=5)
    stud_headers = ["Roll No.", "Group Members", "Email ID", "Contact No.", "Signature"]
    stud_data = [
        ["101", "Student Candidate 1", "candidate1@student.edu", "+91 98765 43210", "_____________"],
        ["102", "Student Candidate 2", "candidate2@student.edu", "+91 98765 43211", "_____________"],
        ["103", "Student Candidate 3", "candidate3@student.edu", "+91 98765 43212", "_____________"],
        ["104", "Student Candidate 4", "candidate4@student.edu", "+91 98765 43213", "_____________"]
    ]
    format_table(stud_table, [0.8, 1.8, 1.9, 1.1, 0.9], stud_headers, stud_data)

    doc.add_paragraph().paragraph_format.space_before = Pt(28)

    # Signatures Block
    sign_table = doc.add_table(rows=2, cols=3)
    sign_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    s_cells = sign_table.rows[0].cells
    s_cells[0].text = "( ___________________ )\nProject Guide\nAssistant Professor, CSE"
    s_cells[1].text = "( ___________________ )\nProject Coordinator\nAssociate Professor, CSE"
    s_cells[2].text = "( ___________________ )\nHead of Department\nDepartment of CSE"
    for cell in s_cells:
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in cell.paragraphs[0].runs:
            r.font.name = 'Times New Roman'
            r.font.size = Pt(10)
            r.bold = True
            r.font.color.rgb = RGBColor(30, 41, 59)

    doc.add_page_break()

    # ==============================================================================
    # ABSTRACT
    # ==============================================================================
    add_main_heading(doc, "ABSTRACT:")
    abstract_text = (
        "CloudPath AI is an artificial intelligence-based weather analysis and monitoring system designed to provide early "
        "meteorological awareness for urban regions, specifically focusing on the Bengaluru Urban Region (12.9716° N, 77.5946° E). "
        "The project is structured into two primary functional modules: Satellite Cloud Movement Tracking and Machine Learning-Based "
        "Rainfall Prediction. In the cloud movement tracking module, geostationary satellite cloud imagery from INSAT-3DS (Level-2B "
        "Cloud Cover products covering July 2025 across 1,448 observations) is extracted over a 162 x 159 pixel grid (~650 km x 650 km). "
        "Digital image processing techniques, including Otsu adaptive thresholding and 8-connectivity Connected Component Analysis (CCA), "
        "are used to detect discrete cloud formations, while the Hungarian optimization algorithm tracks centroid movement between consecutive "
        "30-minute observations to compute speed and directional heading. In the rainfall prediction module, machine learning models "
        "based on Extreme Gradient Boosting (XGBoost) are trained on 96,431 hourly surface meteorological records (2009–2020) comprising "
        "25 atmospheric variables to predict next-hour rain occurrence (classification) and rainfall accumulation in millimeters (regression). "
        "Both modules are unified through a web-based dashboard that displays animated cloud trajectories, motion metrics, and rainfall "
        "predictions. These modules operate using separate datasets and serve as complementary situational awareness tools without claiming "
        "jointly trained multimodal modeling. CloudPath AI enhances localized weather monitoring, supporting urban flood preparedness, "
        "aviation planning, and civic administration."
    )
    add_body_p(doc, abstract_text)
    
    p_kw = doc.add_paragraph()
    p_kw.paragraph_format.space_before = Pt(4)
    p_kw.paragraph_format.space_after = Pt(14)
    r_kwh = p_kw.add_run("Keywords: ")
    r_kwh.bold = True
    r_kwh.font.name = 'Times New Roman'
    r_kwh.font.size = Pt(12)
    r_kwt = p_kw.add_run("Cloud Movement Tracking, INSAT-3DS Satellite, XGBoost, Connected Component Analysis, Rainfall Prediction, Weather Dashboard, Hungarian Algorithm.")
    r_kwt.italic = True
    r_kwt.font.name = 'Times New Roman'
    r_kwt.font.size = Pt(12)

    # ==============================================================================
    # 1. INTRODUCTION
    # ==============================================================================
    add_main_heading(doc, "1. INTRODUCTION:")
    intro_p1 = (
        "Weather monitoring and precipitation forecasting are vital for safeguarding public safety, managing urban infrastructure, "
        "and protecting agricultural productivity. In densely populated metropolitan centers like the Bengaluru Urban Region, localized "
        "convective storm cells frequently develop with rapid intensity, causing sudden waterlogging, traffic gridlocks, and localized "
        "flash flooding. Bengaluru's geographic elevation on the Deccan Plateau (~920 meters above sea level) and its pronounced Urban "
        "Heat Island (UHI) effect foster localized thermodynamic imbalances that can trigger rapid cloud build-up within 30 to 60 minutes."
    )
    add_body_p(doc, intro_p1)

    intro_p2 = (
        "Cloud movement analysis provides early indicators of shifting atmospheric fronts and the approach of rain-bearing storm systems. "
        "By continuously monitoring cloud coverage, displacement vectors, velocity, and formation changes, meteorologists can anticipate "
        "storm trajectories before rainfall begins. Spaceborne geostationary meteorological satellites, such as India's INSAT-3DS operated "
        "by the Indian Space Research Organisation (ISRO), provide regular, high-resolution imagery covering large regional expanses. Concurrently, "
        "advancements in Machine Learning (ML) enable models to learn complex, non-linear relationships from multi-parameter surface weather "
        "telemetry (such as relative humidity, barometric pressure, dew point depression, and solar irradiance). CloudPath AI bridges these "
        "two analytical capabilities, using satellite image processing for cloud motion tracking and machine learning for next-hour rainfall "
        "forecasting, presenting both through an intuitive operational web dashboard."
    )
    add_body_p(doc, intro_p2)

    # ==============================================================================
    # 2. PROBLEM DEFINITION
    # ==============================================================================
    add_main_heading(doc, "2. PROBLEM DEFINITION:")
    prob_p1 = (
        "Traditional weather forecasting relies heavily on large-scale Numerical Weather Prediction (NWP) models and ground-based Doppler "
        "Weather Radars (DWR). While NWP systems are effective for broad regional and multi-day synoptic outlooks, they require extensive "
        "supercomputing resources and exhibit multi-hour data assimilation delays, making them ill-suited for immediate, sub-hourly urban "
        "nowcasting. Ground radar networks, while providing direct hydrometeor reflectivity measurements, face line-of-sight limitations, "
        "ground clutter, and high maintenance costs. Furthermore, manual inspection of satellite imagery by meteorological observers is "
        "labor-intensive and prone to subjective estimation of cloud velocity and trajectory."
    )
    add_body_p(doc, prob_p1)

    prob_p2 = (
        "There is an acute need for an automated computational system that tracks cloud movement across timestamped satellite observations, "
        "calculates objective kinematic parameters (speed, direction, and spatial area), and leverages historical surface weather patterns "
        "to deliver rapid, data-driven next-hour rainfall predictions. CloudPath AI directly addresses this need by providing an automated "
        "solution that combines satellite-based tracking and machine learning inference within a unified, interactive platform."
    )
    add_body_p(doc, prob_p2)

    # ==============================================================================
    # 3. LITERATURE REVIEW
    # ==============================================================================
    add_main_heading(doc, "3. LITERATURE REVIEW:")
    lit_p1 = (
        "A survey of contemporary literature in meteorology and computer science highlights several methodologies:\n"
        "1. Traditional Forecasting: Numerical Weather Prediction frameworks such as the Weather Research and Forecasting (WRF) model integrate "
        "atmospheric fluid dynamics and thermodynamic equations. However, operational run cycles of 3 to 6 hours limit their responsiveness to "
        "rapidly evolving convective clouds.\n"
        "2. Satellite-Based Cloud Monitoring: Geostationary satellites like the INSAT series (INSAT-3D, INSAT-3DR, and the recently deployed INSAT-3DS) "
        "provide half-hourly multi-spectral observations. Level-2B Cloud Cover products generated by MOSDAC offer standardized 4 km pixel grids "
        "suitable for tracking mesoscale convective cloud systems.\n"
        "3. Image Processing for Cloud Tracking: Techniques such as Otsu's adaptive thresholding dynamically separate cloud radiances from surface "
        "backgrounds. Connected Component Analysis (CCA) identifies discrete cloud boundaries, while the Hungarian assignment algorithm (Kuhn, 1955) "
        "solves the bipartite matching problem to maintain track continuity across consecutive frames without greedy search errors.\n"
        "4. Machine Learning for Rainfall Prediction: Tree-based ensemble algorithms, notably Extreme Gradient Boosting (XGBoost; Chen & Guestrin, 2016), "
        "have demonstrated superior accuracy in tabular meteorological forecasting. XGBoost's regularized objective and handling of non-linear "
        "feature interactions make it ideal for addressing the class imbalance typical of rainfall events.\n"
        "5. Integrated Dashboard Presentation: Existing meteorological systems often separate scientific visualization from operational alerts. "
        "An interactive web dashboard that combines spatial trajectories with quantitative KPI metrics is essential for clear communication and timely decision-making."
    )
    add_body_p(doc, lit_p1)

    # ==============================================================================
    # 4. OBJECTIVES AND SCOPE
    # ==============================================================================
    add_main_heading(doc, "4. OBJECTIVES AND SCOPE:")
    obj_p = (
        "The primary objectives of the CloudPath AI project are:\n"
        "• To ingest and preprocess Level-2B NetCDF-4 cloud-mask imagery from ISRO's INSAT-3DS geostationary satellite.\n"
        "• To identify discrete cloud formations and compute their spatial properties (centroids, pixel area, bounding boxes) using digital image processing.\n"
        "• To track cloud formations across consecutive timestamped satellite observations using the Hungarian optimization algorithm.\n"
        "• To calculate quantitative cloud kinematics, including displacement vectors, movement speed (in km/h), and directional heading (in degrees azimuth).\n"
        "• To preprocess historical surface meteorological data (25 atmospheric variables) and engineer robust predictive features.\n"
        "• To train, evaluate, and tune XGBoost models for next-hour rainfall occurrence classification and continuous rainfall amount regression.\n"
        "• To design and deploy an interactive web-based dashboard displaying cloud tracking maps, weather conditions, and dynamic rainfall predictions."
    )
    add_body_p(doc, obj_p)

    scope_p = (
        "Scope of the Project:\n"
        "The current geographical scope is focused on the Bengaluru Urban Region, Karnataka, India (centered at 12.9716° N, 77.5946° E) with an "
        "operational coverage area of 162 x 159 pixels (~650 km x 650 km). Temporally, the satellite tracking module is validated on the full month "
        "of July 2025 INSAT-3DS observations (1,448 frames at ~30-minute intervals). The rainfall prediction models are trained on 96,431 hourly surface "
        "records (2009–2020). These two modules operate as parallel situational awareness streams within a unified dashboard interface."
    )
    add_body_p(doc, scope_p)

    # ==============================================================================
    # 5. PROPOSED METHODOLOGY
    # ==============================================================================
    add_main_heading(doc, "5. PROPOSED METHODOLOGY:")
    meth_p = (
        "The methodology of CloudPath AI follows a structured, step-by-step workflow divided into data acquisition, analytical processing, "
        "and user presentation. The nine primary steps of the workflow are as follows:\n"
        "1. Collect Satellite Cloud-Cover Observations: Retrieve INSAT-3DS Level-2B Cloud Cover NetCDF-4 files at ~30-minute intervals.\n"
        "2. Extract and Preprocess Satellite Data: Verify coordinates and crop the full-disk imagery to a 162 x 159 pixel grid centered on Bengaluru.\n"
        "3. Identify Cloud Regions: Apply Otsu adaptive thresholding to generate binary cloud masks, followed by 3x3 morphological filtering and 8-connectivity Connected Component Analysis (CCA) to isolate discrete cloud objects.\n"
        "4. Track Cloud Formations: Construct a pairwise Euclidean distance cost matrix between centroids of consecutive frames (t-1 and t) and apply the Hungarian optimization algorithm with distance gating (d <= 50 pixels) to associate tracks.\n"
        "5. Calculate Kinematics: Compute cloud velocity components (vx, vy), resultant speed in km/h, and directional heading from centroid displacements and frame timestamps.\n"
        "6. Preprocess Historical Weather Data: Clean, normalize, and temporally partition 96,431 hourly weather records (2009–2020) using an 80/20 train/test split.\n"
        "7. Train and Evaluate Machine Learning Models: Train an XGBoost Classifier for Rain/No Rain classification and an XGBoost Regressor for precipitation depth, optimizing hyperparameters via cross-validation.\n"
        "8. Generate Next-Hour Rainfall Predictions: Execute real-time inference on 25 atmospheric features to produce rain probability and predicted rainfall volume (in mm).\n"
        "9. Display Results Through Dashboard: Render interactive Leaflet/Plotly cloud tracking overlays, kinematic vectors, and dynamic rainfall KPI cards on a web interface."
    )
    add_body_p(doc, meth_p)

    add_diagram(doc, "synopsis_fig2_methodology.png", "Figure 1: Proposed Methodology Workflow of CloudPath AI")

    # ==============================================================================
    # 6. PROPOSED SYSTEM
    # ==============================================================================
    add_main_heading(doc, "6. PROPOSED SYSTEM:")
    sys_intro = (
        "The proposed CloudPath AI system architecture is composed of two independent but complementary modules presented through a unified "
        "operational dashboard interface. It is important to emphasize that both modules currently utilize separate, domain-specific datasets: "
        "the tracking module processes July 2025 satellite observations, while the rainfall prediction module operates on 2009–2020 surface weather data."
    )
    add_body_p(doc, sys_intro)

    add_sub_heading(doc, "A. Satellite Cloud Movement Tracking Module:")
    mod_a_text = (
        "This module processes spaceborne geostationary imagery to provide continuous visual and quantitative tracking of cloud formations. "
        "Its stepwise operational pipeline consists of:\n"
        "• Satellite Image Ingestion: Ingests NetCDF-4 archives, extracting latitude, longitude, and cloud mask matrices.\n"
        "• Cloud-Mask Extraction: Removes missing data codes and applies Otsu thresholding to separate cloud pixels from land surfaces.\n"
        "• Cloud Formation Detection: Labels contiguous 8-connected regions, filtering out small transient noise (< 25 pixels / 400 km²), and calculates geometric centroids (x_bar, y_bar) and bounding boxes.\n"
        "• Cloud Tracking Between Observations: Implements the Hungarian (Kuhn-Munkres) assignment algorithm on a distance-gated cost matrix, maintaining persistent cloud track IDs across successive 30-minute frames.\n"
        "• Direction and Speed Estimation: Derives instantaneous velocity vectors (vx, vy), scalar speed (km/h), and compass heading (degrees azimuth).\n"
        "• Cloud Coverage Visualization: Generates GeoJSON vector layers and heatmaps for timeline scrubbing on the web dashboard."
    )
    add_body_p(doc, mod_a_text)

    add_diagram(doc, "synopsis_fig3_tracking_workflow.png", "Figure 2: Satellite Cloud Movement Tracking Workflow")

    add_sub_heading(doc, "B. Machine Learning-Based Rainfall Prediction Module:")
    mod_b_text = (
        "This module utilizes atmospheric surface telemetry to generate highly accurate, low-latency quantitative precipitation forecasts. "
        "To resolve the zero-inflation characteristic of rainfall (where >80% of hours have zero precipitation), a dual-stage hurdle architecture is used:\n"
        "• Weather Data Preprocessing: Ingests hourly meteorological telemetry and formats a 25-dimensional feature vector encompassing temperature, dew point, relative humidity, barometric pressure, wind vectors, soil moisture, and solar radiation.\n"
        "• Feature Extraction & Scaling: Extracts synoptic and diurnal cyclical features (month, hour) to capture atmospheric thermodynamic trends.\n"
        "• Stage 1 (XGBoost Classification): Evaluates the feature vector to determine the probability of rain occurrence, P(Rain). If P(Rain) < 0.50, the state is classified as 'NO RAIN' and precipitation depth is set to 0.00 mm.\n"
        "• Stage 2 (XGBoost Regression): If P(Rain) >= 0.50, the feature vector is passed to a tuned XGBoost Regressor trained exclusively on non-zero rain events to predict exact continuous rainfall depth in millimeters.\n"
        "• Prediction Result Visualization: Transmits forecast status, rain probability, and predicted depth to the dashboard for instant display."
    )
    add_body_p(doc, mod_b_text)

    add_diagram(doc, "synopsis_fig4_rainfall_workflow.png", "Figure 3: Machine Learning Rainfall Prediction Workflow")

    # ==============================================================================
    # 7. SYSTEM ARCHITECTURE
    # ==============================================================================
    add_main_heading(doc, "7. SYSTEM ARCHITECTURE:")
    arch_intro = (
        "The overall architecture of CloudPath AI connects data sources, analytical algorithms, and user-facing presentation layers into "
        "a clean, decoupled pipeline as illustrated in Figure 4."
    )
    add_body_p(doc, arch_intro)

    add_diagram(doc, "synopsis_fig1_architecture.png", "Figure 4: Overall System Architecture of CloudPath AI")

    arch_exp = (
        "Architecture Explanation:\n"
        "As shown in Figure 4, the architecture comprises two parallel ingestion and processing pipelines converging into a unified presentation layer:\n"
        "1. Satellite Data Pipeline (Top Branch): Raw INSAT-3DS NetCDF-4 imagery is retrieved, cropped to the Bengaluru 162 x 159 pixel domain, "
        "binarized via Otsu's method, and passed to 8-connectivity CCA. Centroids are associated across consecutive frames using the Hungarian "
        "algorithm to estimate speed and directional vectors, which are fed to the dashboard.\n"
        "2. Weather Prediction Pipeline (Bottom Branch): Surface meteorological telemetry is ingested, cleaned, and transformed into a 25-feature "
        "atmospheric vector. The dual-stage XGBoost engine runs classification and regression to compute rain probability and precipitation depth, "
        "pushing the predictions to the dashboard.\n"
        "3. CloudPath AI Dashboard & End User: The Flask-based web application synthesizes the tracking GeoJSON overlays and rainfall predictions, "
        "rendering an interactive, dark-themed operational interface for the meteorologist or urban planner."
    )
    add_body_p(doc, arch_exp)

    # ==============================================================================
    # 8. REQUIREMENT SPECIFICATION
    # ==============================================================================
    add_main_heading(doc, "8. REQUIREMENT SPECIFICATION:")
    
    add_sub_heading(doc, "8.1 Software Requirements:")
    sw_text = (
        "The software components and development libraries utilized in CloudPath AI have been verified against the active codebase:\n"
        "• Operating System: Microsoft Windows 10 / Windows 11 (64-bit).\n"
        "• Database / Storage: No separate external database server (such as MySQL or MongoDB) is required. The system utilizes structured local file-based storage, including NetCDF-4 (.nc) archives, compressed NumPy binary arrays (.npz), CSV files, and serialized Joblib model binaries (.pkl/.joblib).\n"
        "• Hosting: Local hosting via Python HTTP / Flask WSGI server (localhost, default port 8050).\n"
        "• Client Browser: Google Chrome, Microsoft Edge, or any modern standards-compliant web browser.\n"
        "• Code Behind: Python (version 3.10 or higher).\n"
        "• Web Technologies: HTML5, CSS3, JavaScript, Tailwind CSS, Leaflet.js, and Plotly.js.\n"
        "• Libraries & Tools: Pandas, NumPy, XGBoost, Scikit-learn, Matplotlib, SciPy, NetCDF4, OpenCV-Python, and Flask."
    )
    add_body_p(doc, sw_text)

    add_sub_heading(doc, "8.2 Hardware Requirements:")
    hw_text = (
        "The computational hardware required to develop, process, and execute the CloudPath AI system is specified below:\n"
        "• Processor (CPU): Intel Core i3 or above (Intel Core i5 / AMD Ryzen 5 recommended for faster NetCDF-4 decompression).\n"
        "• System Memory (RAM): 8 GB minimum recommended to accommodate in-memory processing of multi-frame satellite image matrices.\n"
        "• Storage (Hard Disk): Minimum 15 GB to 20 GB available storage space to store 31 daily INSAT-3DS satellite ZIP archives, extracted NetCDF-4 files, cached numpy matrices, and trained model weights.\n"
        "• Network Interface: Standard Internet connection required only for initial dataset downloading and map tile streaming."
    )
    add_body_p(doc, hw_text)

    # ==============================================================================
    # 9. PLANNING
    # ==============================================================================
    add_main_heading(doc, "9. PLANNING:")
    plan_intro = (
        "The project development lifecycle for CloudPath AI was structured across eight sequential phases spanning the academic semester. "
        "Table 1 outlines the project planning stages and estimated durations:"
    )
    add_body_p(doc, plan_intro)

    plan_table = doc.add_table(rows=9, cols=4)
    plan_headers = ["Phase", "Project Stage / Milestone", "Estimated Duration", "Key Deliverables"]
    plan_data = [
        ["Phase 1", "Requirement Analysis", "3 Weeks", "Problem formulation, feasibility study, literature review"],
        ["Phase 2", "Data Collection", "3 Weeks", "INSAT-3DS satellite archives (July 2025) & surface weather data (2009-2020)"],
        ["Phase 3", "Data Preprocessing", "4 Weeks", "NetCDF-4 sub-grid cropping (162x159), Otsu mask cleaning, weather feature vector scaling"],
        ["Phase 4", "Model Development", "4 Weeks", "Dual-stage XGBoost model training, hyperparameter tuning, holdout validation"],
        ["Phase 5", "Satellite Cloud Tracking", "3 Weeks", "8-connectivity CCA implementation, Hungarian cost matrix gating, kinematic estimation"],
        ["Phase 6", "Dashboard Development", "3 Weeks", "Midnight Navy UI layout, interactive timeline scrubber, Leaflet map overlays"],
        ["Phase 7", "Integration and Testing", "2 Weeks", "Decoupled pipeline integration, REST endpoint verification, end-to-end testing"],
        ["Phase 8", "Documentation & Submission", "2 Weeks", "Project synopsis, final thesis report, viva presentation materials"]
    ]
    format_table(plan_table, [0.8, 2.2, 1.4, 2.1], plan_headers, plan_data)

    p_tab_cap = doc.add_paragraph()
    p_tab_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_tab_cap.paragraph_format.space_before = Pt(4)
    p_tab_cap.paragraph_format.space_after = Pt(12)
    r_tc = p_tab_cap.add_run("Table 1: Project Planning Stages and Timeline Milestones")
    r_tc.font.name = 'Times New Roman'
    r_tc.font.size = Pt(10.5)
    r_tc.bold = True
    r_tc.font.color.rgb = RGBColor(30, 41, 59)

    # ==============================================================================
    # 10. CONCLUSION & FUTURE SCOPE
    # ==============================================================================
    add_main_heading(doc, "10. CONCLUSION & FUTURE SCOPE:")
    conc_p1 = (
        "CloudPath AI successfully demonstrates the feasibility of combining satellite remote sensing image processing with machine "
        "learning for localized weather awareness. By analyzing 1,448 INSAT-3DS geostationary satellite frames over the Bengaluru Urban "
        "Region for July 2025, the system identifies discrete cloud formations and tracks their trajectories using the Hungarian algorithm, "
        "extracting objective speed and direction kinematics. Concurrently, a dual-stage tuned XGBoost model trained on 12 years of surface "
        "weather telemetry provides accurate next-hour rainfall probability and quantitative accumulation estimates. Unifying these capabilities "
        "in an interactive dashboard provides urban administrators and meteorologists with an effective decision-support tool."
    )
    add_body_p(doc, conc_p1)

    conc_p2 = (
        "Future Scope:\n"
        "While the current implementation achieves its core objectives, future enhancements include:\n"
        "• Real-Time Satellite Data Streaming: Connecting directly to automated MOSDAC / IMD API endpoints for real-time ingestion of newly captured satellite frames as they are downlinked.\n"
        "• Multispectral & Radar Fusion: Incorporating INSAT-3DS Thermal Infrared brightness temperature differences (BTD) and ground Doppler radar reflectivity to improve convective storm intensity estimation.\n"
        "• Synchronized Multimodal Dataset Training: Compiling temporally aligned satellite imagery and weather telemetry over multi-year periods to explore end-to-end deep multimodal learning.\n"
        "• Mobile Application & Push Notifications: Developing a mobile alert client with geofenced push notifications to warn citizens of approaching rainstorms.\n"
        "• Geographical Expansion: Extending the spatial domain and tracking algorithms to other flood-prone urban zones across India."
    )
    add_body_p(doc, conc_p2)

    # ==============================================================================
    # 11. REFERENCES
    # ==============================================================================
    add_main_heading(doc, "11. REFERENCES:")
    ref_items = [
        "[1] T. Chen and C. Guestrin, 'XGBoost: A Scalable Tree Boosting System,' in Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16), San Francisco, CA, USA, 2016, pp. 785–794.",
        "[2] H. W. Kuhn, 'The Hungarian method for the assignment problem,' Naval Research Logistics Quarterly, vol. 2, no. 1‐2, pp. 83–97, Mar. 1955.",
        "[3] N. Otsu, 'A threshold selection method from gray-level histograms,' IEEE Transactions on Systems, Man, and Cybernetics, vol. 9, no. 1, pp. 62–66, Jan. 1979.",
        "[4] Indian Space Research Organisation (ISRO), 'INSAT-3DS Meteorological Satellite Mission Overview and Sensor Payloads,' Meteorological and Oceanographic Satellite Data Archival Centre (MOSDAC), Tech. Rep., 2024. [Online]. Available: https://www.mosdac.gov.in.",
        "[5] India Meteorological Department (IMD), 'Standard Operating Procedure for Weather Forecasting and Nowcasting Services,' Ministry of Earth Sciences, Government of India, Tech. Rep., 2021. [Online]. Available: https://mausam.imd.gov.in.",
        "[6] F. Pedregosa et al., 'Scikit-learn: Machine Learning in Python,' Journal of Machine Learning Research, vol. 12, pp. 2825–2830, Nov. 2011.",
        "[7] Python Software Foundation, 'Python Language Reference and Standard Library Documentation (Version 3.10),' 2023. [Online]. Available: https://www.python.org/doc/.",
        "[8] Open-Meteo, 'Historical Weather API and Meteorological Reanalysis Documentation,' 2023. [Online]. Available: https://open-meteo.com/en/docs/historical-weather-api."
    ]
    for r in ref_items:
        p_r = doc.add_paragraph()
        p_r.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p_r.paragraph_format.space_after = Pt(6)
        p_r.paragraph_format.line_spacing = 1.15
        r_run = p_r.add_run(r)
        r_run.font.name = 'Times New Roman'
        r_run.font.size = Pt(10.5)
        r_run.font.color.rgb = RGBColor(30, 41, 59)

    doc.save(OUT_PATH)
    print(f"Successfully generated Project Synopsis: {OUT_PATH}")

if __name__ == "__main__":
    build_synopsis()
