# ✅ Updated Streamlit UI customization for title & result section
# ⚠️ Detection logic UNCHANGED as requested

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
st.set_page_config(page_title="Blood Cell Detection System", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
        /* Fancy Title */
        .custom-title {
            font-size: 42px;
            font-weight: 900;
            background: linear-gradient(90deg, #b30000, #ff4d4d);
            -webkit-background-clip: text;
            color: transparent;
            text-align: center;
            margin-bottom: 15px;
        }

        /* Fancy result card */
        .result-box {
            background: #fff5f5;
            padding: 20px;
            border-radius: 12px;
            border: 2px solid #b30000;
            text-align: center;
            font-size: 18px;
        }
        .result-number {
            font-size: 32px;
            font-weight: 800;
            color: #b30000;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.markdown("<h1 class='custom-title'>Blood Cell Type Detection & Counting System</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🩸 Blood Detection")
st.sidebar.write("Upload a microscopic blood smear image to detect WBC & RBC cells.")
st.sidebar.write("---")
st.sidebar.write("• Use well-stained microscope images")
st.sidebar.write("• Avoid shadows / overexposure")

# Example Display
if os.path.exists(SAMPLE_PATH):
    sample_img = cv2.cvtColor(cv2.imread(SAMPLE_PATH), cv2.COLOR_BGR2RGB)
    st.image(sample_img, caption="Example Microscopic Blood Sample", use_container_width=True)
else:
    st.warning("Example image not found.")

st.write("Upload a blood smear image to automatically classify **WBC** and **RBC** cells.")

# Upload file
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Invalid image.")
    else:
        st.success("✅ Image uploaded successfully!")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption="Original Image", use_container_width=True)

        # ===== Processing (unchanged) =====
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_purple, upper_purple = np.array([110,60,60]), np.array([160,255,255])
        purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
        purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(purple_mask)
        if num_labels > 1:
            largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            wbc_nucleus_mask = np.uint8(labels == largest_idx) * 255
        else:
            wbc_nucleus_mask = np.zeros_like(purple_mask)

        contours_wbc,_ = cv2.findContours(wbc_nucleus_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        wbc_cells = []
        for contour in contours_wbc:
            area = cv2.contourArea(contour)
            peri = cv2.arcLength(contour, True)
            if 800 < area < 30000 and peri>0:
                if (4*np.pi*area/(peri**2)) > 0.1:
                    wbc_cells.append(contour)

        lower_body, upper_body = np.array([100,25,20]), np.array([170,255,255])
        purple_body = cv2.inRange(hsv, lower_body, upper_body)
        wbc_body = wbc_nucleus_mask.copy()
        grow_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
        for _ in range(10):
            d = cv2.dilate(wbc_body, grow_kernel)
            wbc_body = cv2.bitwise_or(wbc_body, cv2.bitwise_and(d,purple_body))
        wbc_exc = cv2.dilate(wbc_body, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(3.0,(8,8))
        enhanced = clahe.apply(gray)
        blurred = cv2.medianBlur(enhanced,5)
        edges = cv2.Canny(blurred,40,100)
        edges[wbc_exc>0] = 0
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT,1.2,20,param1=80,param2=25,minRadius=10,maxRadius=35)
        rbc_cells = []
        if circles is not None:
            for (x,y,r) in np.round(circles[0,:]).astype(int):
                if wbc_exc[y,x] == 0 and not any(np.hypot(x-a,y-b) < (r+c)*1.0 for (a,b,c) in rbc_cells):
                    rbc_cells.append((x,y,r))

        # Draw results
        result = image_rgb.copy()
        for c in wbc_cells:
            (x,y),r = cv2.minEnclosingCircle(c)
            cv2.circle(result,(int(x),int(y)),int(r*1.4),(0,255,0),3)
            cv2.putText(result,"WBC",(int(x)-25,int(y)-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        for (x,y,r) in rbc_cells:
            cv2.circle(result,(x,y),int(r*1.05),(255,0,0),2)
            cv2.putText(result,"RBC",(x-12,y+4),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,0,0),1)

        total_wbc, total_rbc = len(wbc_cells), len(rbc_cells)
        ratio = total_wbc/total_rbc if total_rbc>0 else 0
        status = "⚠️ Possible infection (high WBC count)" if ratio > 0.02 else "✅ Normal ratio"

        st.image(result, caption=f"Detected Cells", use_container_width=True)

        # ========= Fancy Metrics Box =========
        st.markdown("""
        <div class='result-box'>
            <p><b>White Blood Cells:</b> <span class='result-number'>""" + str(total_wbc) + """</span></p>
            <p><b>Red Blood Cells:</b> <span class='result-number'>""" + str(total_rbc) + """</span></p>
            <p><b>WBC/RBC Ratio:</b> <span class='result-number'>""" + f"{ratio:.4f}" + """</span></p>
            <p>""" + status + """</p>
        </div>
        """, unsafe_allow_html=True)
