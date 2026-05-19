import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Prevents background GUI conflicts on Windows/Streamlit
import matplotlib.pyplot as plt

def analyze_rice(image_bytes):
    # 1. Load and decode the uploaded image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None, "Could not decode image. Please upload a valid image file."
        
    original = img.copy()
    h_img, w_img, _ = img.shape
    total_img_area = h_img * w_img
    
    # 2. Preprocessing (Grayscale & Blur to remove camera noise)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive Thresholding isolates the grains completely away from a dark tray
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 51, -15
    )
    
    # Morphological Opening cleans up stray background specks
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 3. Distance Transform & Watershed Segmentation (Cuts touching grains apart)
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
    
    # --- PHASE 1: PRE-SCAN TO CALCULATE CAMERA CONVERSION SCALE ---
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
        
        # Filter dust spots and massive multi-grain clusters
        if area < 30 or area > (total_img_area * 0.01):
            continue
            
        rect = cv2.minAreaRect(c)
        (_, _), (w, h), _ = rect
        length_px = max(w, h)
        initial_pixel_lengths.append(length_px)
        
    if not initial_pixel_lengths:
        return None, "No valid rice grains detected. Check lighting or background contrast."

    # Establish pixel-to-mm conversion ratio baseline (Assuming 6.5 mm per normal grain)
    median_px_length = np.median(initial_pixel_lengths)
    mm_per_pixel = 6.5 / median_px_length

    # --- PHASE 2: DETAILED SCAN & GHOST BOX BORDER PURGE ---
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
        
        # Convert to real-world physical metric units (mm)
        length_mm = length_px * mm_per_pixel
        
        # CRITICAL GUARDRAIL: Automatically eject the "Ghost Box" tray border lines.
        # No actual individual rice grain will exceed 15mm in size.
        if length_mm > 15.0:
            continue
            
        grain_mm_lengths.append(length_mm)
        valid_contours.append((c, length_mm))

    if not grain_mm_lengths:
        return None, "All detected objects were classified as tray border artifacts. Keep rice away from edges."

    # 4. Statistical Rice Quality Analysis calculations
    median_mm = np.median(grain_mm_lengths)
    threshold_mm = median_mm * 0.75  # Pieces shorter than 75% of average size are marked broken
    
    total_grains = len(grain_mm_lengths)
    full_count = sum(1 for l in grain_mm_lengths if l >= threshold_mm)
    broken_count = total_grains - full_count
    broken_percentage = (broken_count / total_grains) * 100
    
    # 5. Render final visual classification image overlay
    output_img = original.copy()
    for c, length_mm in valid_contours:
        # Green for Full, Red for Broken
        color = (0, 255, 0) if length_mm >= threshold_mm else (0, 0, 255)
        cv2.drawContours(output_img, [c], -1, color, 2)

    results = {
        "Total Grains": total_grains,
        "Full Grains": full_count,
        "Broken Grains": broken_count,
        "Broken Percentage": f"{broken_percentage:.2f}%",
        "Average Length (mm)": f"{np.mean(grain_mm_lengths):.2f} mm",
        "Max Length (mm)": f"{np.max(grain_mm_lengths):.2f} mm",
        "Min Length (mm)": f"{np.min(grain_mm_lengths):.2f} mm",
        "Output Image": output_img
    }
    return results, None

# --- STREAMLIT USER INTERFACE (WEB DISPLAY) ---
st.set_page_config(page_title="AI Rice Quality Analyzer", layout="wide")
st.title("🌾 AI Rice Grain Length & Quality Analyzer")
st.write("Upload a clear picture of rice grains spread out evenly on a contrasting dark background surface.")

uploaded_file = st.file_uploader("Choose a rice sample image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    with st.spinner("Processing image and filtering tray boundaries..."):
        results, error = analyze_rice(file_bytes)
        
    if error:
        st.error(error)
    else:
        # Setup clean two-column grid layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Analysis Visualization")
            st.image(results["Output Image"], channels="BGR", use_container_width=True)
            st.caption("🟢 Green: Full Grains | 🔴 Red: Broken Grains (Tray border lines successfully omitted)")
            
        with col2:
            st.subheader("Quality Metrics Dashboard (Metric System)")
            
            metrics_df = pd.DataFrame({
                "Metric": [
                    "Total Grains Tracked", 
                    "Full Grains Count", 
                    "Broken Grains Count", 
                    "Broken Percentage", 
                    "Average Grain Length",
                    "Maximum Grain Length",
                    "Minimum Grain Length"
                ],
                "Value": [
                    results["Total Grains"], 
                    results["Full Grains"], 
                    results["Broken Grains"], 
                    results["Broken Percentage"], 
                    results["Average Length (mm)"],
                    results["Max Length (mm)"],
                    results["Min Length (mm)"]
                ]
            })
            
            st.table(metrics_df)
            st.info(f"💡 **Analysis Verdict:** This batch contains **{results['Broken Percentage']}** broken rice particles.")