import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
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

# ================= Custom CSS =================
st.markdown(
    """
    <style>
    .big-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg,#ff4b4b,#ff8b8b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 12px;
    }
    .result-box {
        background: rgba(255,255,255,0.08);
        padding: 20px;
        border-radius: 15px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="big-title">🩸 Blood Cell Detection & Counting System</div>', unsafe_allow_html=True)

st.write("Upload a microscopic blood smear image to detect and count RBC and WBC cells.")

# Sidebar summary placeholder
stats_placeholder = st.sidebar.empty()
st.sidebar.write("---")
st.sidebar.info("Upload an image to start analysis.")

# ============================================================
# File Upload
# ============================================================
uploaded_file = st.file_uploader("📤 Upload Blood Cell Image", type=["jpg", "jpeg", "png"])

# Show sample
if os.path.exists(SAMPLE_PATH):
    st.image(cv2.cvtColor(cv2.imread(SAMPLE_PATH), cv2.COLOR_BGR2RGB), use_container_width=True, caption="Sample Microscopic Image")

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Could not read the uploaded image.")
    else:
        st.success("✅ Image uploaded successfully!")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        tab1, tab2, tab3 = st.tabs(["🖼 Original Image", "🔍 Processed Output", "📊 Statistics & Report"])

        with tab1:
            st.image(image_rgb, caption="Uploaded Image", use_container_width=True)

        with st.spinner("🔬 Analyzing image, please wait..."):
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # WBC detection
            lower_purple = np.array([110, 60, 60]); upper_purple = np.array([160, 255, 255])
            purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(purple_mask)
            if num_labels > 1:
                largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                wbc_nucleus_mask = np.uint8(labels == largest_idx) * 255
            else:
                wbc_nucleus_mask = np.zeros_like(purple_mask)
            contours_wbc, _ = cv2.findContours(wbc_nucleus_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            wbc_cells = []
            for contour in contours_wbc:
                area = cv2.contourArea(contour)
                if 800 < area < 30000:
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter ** 2)
                        if circularity > 0.1:
                            wbc_cells.append(contour)

            # RBC detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            blurred = cv2.medianBlur(enhanced, 5)
            circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.2, 20, param1=80, param2=25, minRadius=10, maxRadius=35)
            rbc_cells = []
            if circles is not None:
                circles = np.round(circles[0, :]).astype(int)
                for (x, y, r) in circles:
                    too_close = any(np.hypot(x - fx, y - fy) < (r + fr) * 1.0 for (fx, fy, fr) in rbc_cells)
                    if not too_close:
                        rbc_cells.append((x, y, r))

            total_wbc = len(wbc_cells)
            total_rbc = len(rbc_cells)
            ratio = total_wbc / total_rbc if total_rbc > 0 else 0
            status = "🟡 Possible infection (high WBC count)" if ratio > 0.02 else "🟢 Normal cell ratio"

            result = image_rgb.copy()
            for contour in wbc_cells:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                cv2.circle(result, (int(x), int(y)), int(radius * 1.4), (0, 255, 0), 3)
                cv2.putText(result, "WBC", (int(x) - 25, int(y) - int(radius) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            for (x, y, r) in rbc_cells:
                cv2.circle(result, (x, y), int(r * 1.05), (0, 0, 255), 2)
                cv2.putText(result, "RBC", (x - 12, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        with tab2:
            st.image(result, caption=f"WBC: {total_wbc} | RBC: {total_rbc}", use_container_width=True)
            st.success("✅ Detection complete!")

        with tab3:
            st.markdown(f"<div class='result-box'><b>White Blood Cells:</b> {total_wbc}<br><b>Red Blood Cells:</b> {total_rbc}<br><b>WBC/RBC Ratio:</b> {ratio:.4f}<br><b>Status:</b> {status}</div>", unsafe_allow_html=True)

            st.metric("WBC", total_wbc)
            st.metric("RBC", total_rbc)
            st.metric("WBC/RBC Ratio", f"{ratio:.4f}")

        stats_placeholder.metric("WBC", total_wbc)
        stats_placeholder.metric("RBC", total_rbc)
        stats_placeholder.metric("WBC/RBC Ratio", f"{ratio:.4f}")
        st.balloons()

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        df = pd.DataFrame({"Cell Type": ["WBC", "RBC"], "Count": [total_wbc, total_rbc], "Ratio": [ratio, ""]})

        st.download_button("⬇ Download Annotated Image", data=buffer.tobytes(), file_name=f"blood_result_{timestamp}.jpg", mime="image/jpeg")
        st.download_button("⬇ Download CSV Report", data=df.to_csv(index=False).encode(), file_name=f"blood_stats_{timestamp}.csv", mime="text/csv")

else:
    st.info("📥 Upload a blood image to start analysis.")
