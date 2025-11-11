## blood-cell-detection-streamlit
Streamlit-based blood cell detection system that analyzes microscopic blood smear images to identify and count red blood cells (RBCs) and white blood cells (WBC)s.

---

## Overview:
This application automates the detection and counting of blood cells using color segmentation and classical image processing techniques in OpenCV.
The system identifies RBCs and WBCs based on their distinct color characteristics — RBCs appear in light pink to red hues, while WBCs exhibit purple to blue hues due to nucleus staining.

It provides real-time results with visual annotations, total cell counts, WBC/RBC ratio, and downloadable reports — all within an interactive Streamlit interface.

---

## Features:
- Upload and analyze microscopic blood smear images.
- Detect and label RBCs and WBCs using HSV color segmentation.
- Display WBC/RBC ratio and flag potential infection cases.
- Download annotated image, statistics (CSV), and text report.
- Simple, interactive web interface built with Streamlit.

---

## Installation:
1. *Clone the repository*
   ```bash
   git clone https://github.com/<your-username>/blood-cell-detection-streamlit.git
   cd blood-cell-detection-streamlit

2. *Install dependencies*
   ```bash
   pip install -r requirements.txt

3. *Run the Streamlit app*
   ```bash
   streamlit run app.py

---

## Requirements:
Make sure your requirements.txt includes:
- streamlit
- opencv-python
- numpy
- pandas

---

## How It Works:
1. Convert uploaded image to HSV color space for better color segmentation.
2. Detect WBCs using purple color thresholds and morphological filtering.
3. Detect RBCs using the Hough Circle Transform after preprocessing.
4. Annotate results and calculate WBC/RBC ratio to estimate possible infection.

---

## Interpretation of Results:
The app calculates the WBC/RBC ratio, which provides insight into potential abnormalities in blood samples.

In small image samples, the ratio becomes highly sensitive to even one detected WBC — for example, one WBC among 18 RBCs gives a ratio of 0.0556, which can misleadingly appear abnormal despite being within normal biological range when viewed across the entire sample.

This emphasizes that small sample images may not represent the actual clinical ratio, and variations in staining or lighting can also affect color segmentation accuracy.

---

## Project Structure:

├── app.py                           # Main Streamlit application

├── requirements.txt                 # List of dependencies

├── sample_bloodcell_streamlit.jpg   # Example image for user reference

└── README.md                        # Documentation

---

## The app displays:
- Original uploaded image
- Annotated detection results with labeled RBCs and WBCs
- WBC/RBC ratio and infection status
- Downloadable report files

---

## Online Access:
Try the app live here: [Automated Blood Cell Detection](https://blood-cell-detection-app-3opttkwuzhug33esarbho9.streamlit.app/)


   
