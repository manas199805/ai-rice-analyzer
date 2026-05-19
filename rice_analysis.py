import streamlit as st
import cv2
import numpy as np
import pandas as pd
import urllib.parse
import random
import io  # NEW: Added for direct in-memory data processing
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

# ReportLab core engines
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def analyze_rice(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None, "Could not decode image."
        
    original = img.copy()
    h_img, w_img, _ = img.shape
    total_img_area = h_img * w_img
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 51, -15
    )
    
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    sure_bg = cv2.dilate(opening, kernel, iterations=2)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    markers = cv2.watershed(img, markers)
    unique_labels = np.unique(markers)
    
    initial_pixel_lengths = []
    for label in unique_labels:
        if label <= 1:
            continue
        grain_mask = np.zeros(gray.shape, dtype="uint8")
        grain_mask[markers == label] = 255
        contours, _ = cv2.findContours(grain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < 30 or area > (total_img_area * 0.01):
            continue
        rect = cv2.minAreaRect(c)
        (_, _), (w, h), _ = rect
        length_px = max(w, h)
        initial_pixel_lengths.append(length_px)
        
    if not initial_pixel_lengths:
        return None, "No valid rice grains detected."

    median_px_length = np.median(initial_pixel_lengths)
    mm_per_pixel = 6.5 / median_px_length

    grain_mm_lengths = []
    valid_contours = []
    for label in unique_labels:
        if label <= 1:
            continue
        grain_mask = np.zeros(gray.shape, dtype="uint8")
        grain_mask[markers == label] = 255
        contours, _ = cv2.findContours(grain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < 30:
            continue
        rect = cv2.minAreaRect(c)
        (_, _), (w, h), _ = rect
        length_px = max(w, h)
        length_mm = length_px * mm_per_pixel
        if length_mm > 15.0:
            continue
        grain_mm_lengths.append(length_mm)
        valid_contours.append((c, length_mm))

    median_mm = np.median(grain_mm_lengths)
    threshold_mm = median_mm * 0.75  
    
    total_grains = len(grain_mm_lengths)
    full_count = sum(1 for l in grain_mm_lengths if l >= threshold_mm)
    broken_count = total_grains - full_count
    broken_percentage = (broken_count / total_grains) * 100
    
    output_img = original.copy()
    for c, length_mm in valid_contours:
        color = (0, 255, 0) if length_mm >= threshold_mm else (0, 0, 255)
        cv2.drawContours(output_img, [c], -1, color, 2)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(output_img, f"{length_mm:.1f}", (cX - 12, cY + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

    results = {
        "Total Grains": total_grains,
        "Full Grains": full_count,
        "Broken Grains": broken_count,
        "Broken Percentage": f"{broken_percentage:.2f}%",
        "Average Length": f"{np.mean(grain_mm_lengths):.2f} mm",
        "Max Length": f"{np.max(grain_mm_lengths):.2f} mm",
        "Min Length": f"{np.min(grain_mm_lengths):.2f} mm",
        "Output Image": output_img
    }
    return results, None

def generate_pdf_report(results):
    # Create an in-memory buffer for the PDF document instead of a physical file
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=22, textColor=colors.HexColor('#2b6cb0'), spaceAfter=5)
    subtitle_style = ParagraphStyle('SubTitleStyle', parent=styles['Normal'], fontSize=10, textColor=colors.HexColor('#4a5568'), spaceAfter=20)
    section_style = ParagraphStyle('SectionStyle', parent=styles['Heading2'], fontSize=12, textColor=colors.HexColor('#2d3748'), spaceBefore=15, spaceAfter=8)
    
    story.append(Paragraph("Rice Quality Inspection Report", title_style))
    story.append(Paragraph(f"Computer Vision & Metric Sizing Systems | Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", subtitle_style))
    
    verdict_text = f"<b>Analysis Summary Verdict:</b> This batch contains <b>{results['Broken Percentage']}</b> broken components out of <b>{results['Total Grains']}</b> total tracked rice grains."
    verdict_style = ParagraphStyle('VerdictStyle', parent=styles['Normal'], fontSize=11, textColor=colors.HexColor('#2c5282'))
    
    verdict_table = Table([[Paragraph(verdict_text, verdict_style)]], colWidths=[530])
    verdict_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#ebf8ff')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#3182ce')),
        ('PADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(verdict_table)
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("Metrics Specifications", section_style))
    data = [
        [Paragraph("<b>Quality Metric Parameter</b>", styles['Normal']), Paragraph("<b>Inspected Value</b>", styles['Normal'])],
        ["Total Grains Tracked", str(results['Total Grains'])],
        ["Full Grains Count", str(results['Full Grains'])],
        ["Broken Grains Count", str(results['Broken Grains'])],
        ["Broken Percentage Ratio", str(results['Broken Percentage'])],
        ["Average Grain Length", str(results['Average Length'])],
        ["Maximum Grain Length", str(results['Max Length'])],
        ["Minimum Grain Length", str(results['Min Length'])],
    ]
    
    metrics_table = Table(data, colWidths=[265, 265])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f7fafc')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e0')),
        ('PADDING', (0,0), (-1,-1), 6),
        ('BACKGROUND', (0,4), (-1,4), colors.HexColor('#edf2f7')), 
        ('FONTNAME', (0,4), (-1,4), 'Helvetica-Bold'),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 15))
    
    # FIX: Write the image into memory instead of saving to a file on your hard drive
    _, img_encoded = cv2.imencode('.jpg', results["Output Image"])
    img_io = io.BytesIO(img_encoded.tobytes())
    
    story.append(Paragraph("Processed Image Diagnostic Layout", section_style))
    # ReportLab safely reads directly out of the memory buffer object now!
    story.append(Image(img_io, width=320, height=320 * (results["Output Image"].shape[0] / results["Output Image"].shape[1])))
    
    doc.build(story)
    
    # Retrieve data straight out of the memory address stream
    pdf_bytes = pdf_buffer.getvalue()
    pdf_buffer.close()
    return pdf_bytes

# --- STREAMLIT USER INTERFACE ---
st.set_page_config(page_title="AI Rice Quality Analyzer", layout="wide")
st.title("🌾 AI Rice Grain Length & Quality Analyzer")
st.write("Upload a clear picture of rice grains spread out evenly on a contrasting dark background surface.")

uploaded_file = st.file_uploader("Choose a rice sample image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    with st.spinner("Processing image and executing structural segmentation..."):
        results, error = analyze_rice(file_bytes)
        
    if error:
        st.error(error)
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Analysis Visualization")
            st.image(results["Output Image"], channels="BGR", use_container_width=True)
            st.caption("🟢 Green: Full Grains | 🔴 Red: Broken Grains | White Numbers: Sizing in mm")
            
            _, img_encoded = cv2.imencode('.jpg', results["Output Image"])
            st.download_button(
                label="📥 Download Result Image File",
                data=img_encoded.tobytes(),
                file_name="rice_analysis_report.jpg",
                mime="image/jpeg",
                use_container_width=True
            )
            
        with col2:
            st.subheader("Quality Metrics Dashboard")
            
            metrics_df = pd.DataFrame({
                "Metric": [
                    "Total Grains Tracked", "Full Grains Count", "Broken Grains Count", 
                    "Broken Percentage", "Average Grain Length", "Maximum Grain Length", "Minimum Grain Length"
                ],
                "Value": [
                    results["Total Grains"], results["Full Grains"], results["Broken Grains"], 
                    results["Broken Percentage"], results["Average Length"], results["Max Length"], results["Min Length"]
                ]
            })
            st.table(metrics_df)
            
            st.subheader("📄 Export Formal Documents")
            
            # App builds the PDF completely in system RAM on request
            st.download_button(
                label="📥 Download Official PDF Quality Certificate",
                data=generate_pdf_report(results),
                file_name=f"Rice_Inspection_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            
            report_text = (
                f"🌾 *AI RICE QUALITY REPORT*\n\n"
                f"📊 *Summary Metrics:*\n"
                f"• Total Grains: {results['Total Grains']}\n"
                f"• Full Grains: {results['Full Grains']}\n"
                f"• Broken Grains: {results['Broken Grains']}\n"
                f"• *Broken Percentage: {results['Broken Percentage']}*\n\n"
                f"📏 *Sizing Profile:*\n"
                f"• Avg Length: {results['Average Length']}\n"
                f"• Max Length: {results['Max Length']}\n"
                f"• Min Length: {results['Min Length']}"
            )
            encoded_report = urllib.parse.quote(report_text)
            whatsapp_url = f"https://api.whatsapp.com/send?text={encoded_report}"
            
            st.markdown(
                f'''
                <a href="{whatsapp_url}" target="_blank" style="text-decoration: none;">
                    <div style="background-color: #25D366; color: white; text-align: center; 
                    padding: 10px; border-radius: 8px; font-weight: bold; font-size: 16px; margin-top: 15px;">
                        💬 Share Text Summary via WhatsApp
                    </div>
                </a>
                ''', 
                unsafe_allow_html=True
            )