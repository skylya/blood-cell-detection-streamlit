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

# ================= Custom CSS (Design Only, Logic Untouched) =================
st.markdown(
    """
    <style>
    .title {
        font-size: 40px;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg,#d60000,#ff8b8b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    .subtitle {
        text-align:center;
        font-size: 18px;
        color:#444;
        margin-bottom:20px;
    }
    .result-box{
        border-radius:12px;
        padding:18px;
        background:rgba(255,245,245,0.9);
        border:1px solid #ffb3b3;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">🩸 Blood Cell Type Detection & Counting</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a microscopic blood smear image for automated WBC & RBC detection.</div>', unsafe_allow_html=True)

# Sidebar info
st.sidebar.title("📌 Info Panel")
st.sidebar.write("Upload image to begin analysis")

# Example image
if os.path.exists(SAMPLE_PATH):
    sample_img = cv2.cvtColor(cv2.imread(SAMPLE_PATH), cv2.COLOR_BGR2RGB)
    st.image(sample_img, caption="Sample Blood Smear Image", use_container_width=True)
else:
    st.warning("Sample image not found. Upload your own.")

st.info("Upload blood smear image to detect **WBC** & **RBC**")

# File Upload
uploaded_file = st.file_uploader("📤 Upload Blood Cell Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("❌ Failed to read image")
    else:
        st.success("✅ Image uploaded successfully")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        tabs = st.tabs(["🖼 Original Image", "🔬 Processing", "📊 Results"])
        with tabs[0]: st.image(image_rgb, caption="Original Image", use_container_width=True)

        # ================= Processing (Original Logic) =================
        with st.spinner("Analyzing image, please wait..."):
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

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
                    if perimeter>0:
                        circularity = 4*np.pi*area/(perimeter**2)
                        if circularity>0.1: wbc_cells.append(contour)

            lower_purple_body = np.array([100,25,20]); upper_purple_body = np.array([170,255,255])
            purple_body_mask = cv2.inRange(hsv, lower_purple_body, upper_purple_body)
            wbc_body = wbc_nucleus_mask.copy()
            growth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
            for _ in range(10):
                dilated = cv2.dilate(wbc_body,growth_kernel,1)
                dilated = cv2.bitwise_and(dilated,purple_body_mask)
                wbc_body = cv2.bitwise_or(wbc_body,dilated)
            wbc_exclusion = cv2.dilate(wbc_body,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)),1)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(3.0,(8,8))
            enhanced = clahe.apply(gray)
            blurred = cv2.medianBlur(enhanced,5)
            edges = cv2.Canny(blurred,40,100)
            edges[wbc_exclusion>0]=0
            circles = cv2.HoughCircles(blurred,cv2.HOUGH_GRADIENT,1.2,20,param1=80,param2=25,minRadius=10,maxRadius=35)
            rbc_candidates=[]
            if circles is not None:
                circles=np.round(circles[0,:]).astype(int)
                wbc_margin=cv2.dilate(wbc_exclusion,np.ones((15,15),np.uint8),1)
                for (x,y,r) in circles:
                    if wbc_margin[y,x]==0: rbc_candidates.append((x,y,r))
            rbc_cells=[]
            for (x,y,r) in sorted(rbc_candidates,key=lambda c:(c[1],c[0])):
                too_close = any(np.hypot(x-fx,y-fy)<(r+fr) for (fx,fy,fr) in rbc_cells)
                if not too_close: rbc_cells.append((x,y,r))

            result=image_rgb.copy()
            for contour in wbc_cells:
                (x,y),radius=cv2.minEnclosingCircle(contour)
                cv2.circle(result,(int(x),int(y)),int(radius*1.4),(0,255,0),3)
                cv2.putText(result,"WBC",(int(x)-25,int(y)-int(radius)-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            for (x,y,r) in rbc_cells:
                cv2.circle(result,(x,y),int(r*1.05),(0,0,255),2)
                cv2.putText(result,"RBC",(x-12,y+4),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1)

            total_wbc=len(wbc_cells); total_rbc=len(rbc_cells)
            ratio=total_wbc/total_rbc if total_rbc>0 else 0
            status="Possible infection (high WBC count)" if ratio>0.02 else "Normal ratio"

        with tabs[1]: st.image(result,caption=f"WBC: {total_wbc} | RBC: {total_rbc}")
        with tabs[2]:
            st.markdown(
                f"""
                <div style='background:#fff7f7;border:2px solid #ffbdbd;padding:20px;border-radius:14px;box-shadow:0 4px 10px rgba(0,0,0,0.1);margin-bottom:18px;'>
                    <h3 style='color:#b30000;font-weight:800;'>📊 Final Blood Cell Analysis</h3>
                    <p style='font-size:15px;'>Below are the detected cell counts and diagnostic ratio.</p>
                    <hr style='border:1px solid #ffc2c2;'>
                    <b>White Blood Cells (WBC):</b> {total_wbc}<br>
                    <b>Red Blood Cells (RBC):</b> {total_rbc}<br>
                    <b>WBC/RBC Ratio:</b> {ratio:.4f}<br>
                    <b>Status:</b> {status}
                </div>
                """,
                unsafe_allow_html=True,
            )

            c1, c2, c3 = st.columns(3)
            with c1: st.metric("White Blood Cells", total_wbc)
            with c2: st.metric("Red Blood Cells", total_rbc)
            with c3: st.metric("WBC/RBC Ratio", f"{ratio:.4f}")

        timestamp=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        _,buffer=cv2.imencode(".jpg",cv2.cvtColor(result,cv2.COLOR_RGB2BGR))
        df=pd.DataFrame({"Type":["WBC","RBC"],"Count":[total_wbc,total_rbc],"Ratio":[ratio,""]})
        st.download_button("Download Image",buffer.tobytes(),f"blood_result_{timestamp}.jpg","image/jpeg")
        st.download_button("Download CSV",df.to_csv(index=False).encode(),f"blood_stats_{timestamp}.csv","text/csv")("Download Image",buffer.tobytes(),f"blood_result_{timestamp}.jpg","image/jpeg")
        st.download_button("Download CSV",df.to_csv(index=False).encode(),f"blood_stats_{timestamp}.csv","text/csv")

else:
    st.info("📥 Upload a blood image to begin analysis.")
