import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import base64
import os

# ============================================================
# File Path Setup
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PATH = os.path.join(BASE_DIR, "sample_bloodcell_streamlit.jpg")

# ============================================================
# Streamlit App Configuration
# ============================================================
st.set_page_config(page_title="🩸 Blood Cell Detection System", layout="wide")

# Title with color change only 
st.markdown(
    """
    <h1 style='color:#b30000; font-weight:700;'>Blood Cell Type Detection and Counting System</h1>
    """,
    unsafe_allow_html=True,
)

# Sidebar 
st.sidebar.title("🩸 Blood Detection")
st.sidebar.write("Upload a microscopic blood smear image on the main page to automatically detect and classify **WBC** and **RBC** cells.")
st.sidebar.write("---")
st.sidebar.write("Tips:")
st.sidebar.write("- Use well-stained microscopic images")
st.sidebar.write("- Avoid extreme lighting/shadows")

# Safely display example image
if os.path.exists(SAMPLE_PATH):
    sample_img = cv2.cvtColor(cv2.imread(SAMPLE_PATH), cv2.COLOR_BGR2RGB)
    st.image(sample_img, caption="Example of a Microscopic Blood Smear Image", use_container_width=True)
else:
    st.warning("Example image not found. Please upload your own image to continue.")

st.markdown("""
Upload a blood smear image to automatically detect and classify **WBC** and **RBC** cells.
""")

st.markdown("""
> **Important:**  
> For best accuracy, please upload **microscopic blood smear images** similar to the example above.  
> This system uses **color segmentation** to distinguish:
> - **Red Blood Cells (RBCs):** light pink to red hues  
> - **White Blood Cells (WBCs):** purple to blue hues nucleus  
>
> Images with **different lighting, colors, or non-microscopic samples** may produce inaccurate results,  
> as the model is calibrated for this specific staining and microscopy type.
""")

# ============================================================
# File Upload
# ============================================================
uploaded_file = st.file_uploader("Upload Blood Cell Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Could not read the uploaded image.")
    else:
        st.success("Image uploaded successfully!")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption="Original Image", use_container_width=True)

        # ============================================================
        # PROCESSING SECTION
        # ============================================================
        with st.spinner("Analyzing image, please wait..."):
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # --- Purple mask (WBC nucleus) ---
            lower_purple = np.array([110, 60, 60])
            upper_purple = np.array([160, 255, 255])
            purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            # Keep largest blob (WBC nucleus)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(purple_mask)
            if num_labels > 1:
                largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                wbc_nucleus_mask = np.uint8(labels == largest_idx) * 255
            else:
                wbc_nucleus_mask = np.zeros_like(purple_mask)

            # --- Find WBC contours ---
            contours_wbc, _ = cv2.findContours(wbc_nucleus_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_wbc_area, max_wbc_area = 800, 30000
            wbc_cells = []
            for contour in contours_wbc:
                area = cv2.contourArea(contour)
                if min_wbc_area < area < max_wbc_area:
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter ** 2)
                        if circularity > 0.1:
                            wbc_cells.append(contour)

            # --- WBC body expansion ---
            lower_purple_body = np.array([100, 25, 20])
            upper_purple_body = np.array([170, 255, 255])
            purple_body_mask = cv2.inRange(hsv, lower_purple_body, upper_purple_body)
            wbc_body = wbc_nucleus_mask.copy()
            growth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            for _ in range(10):
                dilated = cv2.dilate(wbc_body, growth_kernel, iterations=1)
                dilated = cv2.bitwise_and(dilated, purple_body_mask)
                wbc_body = cv2.bitwise_or(wbc_body, dilated)
            wbc_exclusion = cv2.dilate(wbc_body, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)), iterations=1)

            # --- RBC Detection ---
            gray_rbc = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray_rbc)
            blurred = cv2.medianBlur(enhanced, 5)
            edges = cv2.Canny(blurred, 40, 100)
            edges[wbc_exclusion > 0] = 0

            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                param1=80, param2=25, minRadius=10, maxRadius=35
            )
            rbc_candidates = []
            if circles is not None:
                circles = np.round(circles[0, :]).astype(int)
                wbc_margin = cv2.dilate(wbc_exclusion, np.ones((15, 15), np.uint8), iterations=1)
                for (x, y, r) in circles:
                    if wbc_margin[y, x] == 0:
                        rbc_candidates.append((x, y, r))
            rbc_cells = []
            for (x, y, r) in sorted(rbc_candidates, key=lambda c: (c[1], c[0])):
                too_close = any(np.hypot(x - fx, y - fy) < (r + fr) * 1.0 for (fx, fy, fr) in rbc_cells)
                if not too_close:
                    rbc_cells.append((x, y, r))

            # ============================================================
            # DRAW RESULTS
            # ============================================================
            result = image_rgb.copy()
            for contour in wbc_cells:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                cv2.circle(result, (int(x), int(y)), int(radius * 1.4), (0, 255, 0), 3)
                cv2.putText(result, "WBC", (int(x) - 25, int(y) - int(radius) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            for (x, y, r) in rbc_cells:
                cv2.circle(result, (x, y), int(r * 1.05), (0, 0, 255), 2)
                cv2.putText(result, "RBC", (x - 12, y + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # ============================================================
            # STATS + REPORT
            # ============================================================
            total_wbc = len(wbc_cells)
            total_rbc = len(rbc_cells)
            total_cells = total_wbc + total_rbc
            ratio = total_wbc / total_rbc if total_rbc > 0 else 0
            status = "Possible infection (high WBC count)" if ratio > 0.02 else "Normal ratio"

            # ============================================================
            # DISPLAY RESULTS 
            # ============================================================
            
            tabs = st.tabs(["Processed Image", "Cell Count Summary"])
            
            with tabs[0]:
                st.image(result, caption=f"WBC: {total_wbc} | RBC: {total_rbc}", use_container_width=True)
            
            with tabs[1]:
                st.header("Detection Summary")
                
                st.metric("White Blood Cells (WBC)", total_wbc)
                st.metric("Red Blood Cells (RBC)", total_rbc)
                st.metric("WBC / RBC Ratio", f"{ratio:.4f}")
            
                st.write("---")
                
                if ratio > 0.02:
                    st.error("Possible infection (high WBC count)")
                else:
                    st.success("Normal WBC/RBC ratio")
            
                st.write(f"**Total Cells Detected:** {total_cells}")
            





