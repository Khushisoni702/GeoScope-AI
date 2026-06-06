import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import plotly.express as px
from reportlab.pdfgen import canvas
from io import BytesIO
from datetime import datetime
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="GeoScope AI",
    layout="wide"
)

# ----------------------------
# CUSTOM CSS
# ----------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.main {
    background-color: #F7F9FC;
}

.block-container {
    padding-top: 1rem;
    max-width: 1200px;
}

.hero {
    position: relative;
    background-image:
        linear-gradient(
            rgba(0,0,0,0.60),
            rgba(0,0,0,0.60)
        ),
        url("https://images.unsplash.com/photo-1462331940025-496dfbfc7564");

    background-size: cover;
    background-position: center;
    border-radius: 24px;
    padding: 90px 60px;
    margin-bottom: 35px;
    text-align: center;
}

.hero-title {
    color: white;
    font-size: 64px;
    font-weight: 700;
    letter-spacing: 1px;
}

.hero-subtitle {
    color: rgba(255,255,255,0.95);
    font-size: 24px;
    margin-top: 10px;
}

.result-card {
    background: transparent;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.08);
}

.prediction-text {
    font-size: 30px;
    font-weight: 600;
    color: #1f2937;
}

.confidence-text {
    font-size: 20px;
    color: #4b5563;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# CLASS NAMES
# ----------------------------
class_names = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]

# ----------------------------
# LAND USE DESCRIPTIONS
# ----------------------------
descriptions = {
    "AnnualCrop": "Agricultural regions cultivated with crops that are planted and harvested within a single growing season. These areas are essential for food production and are characterized by regularly patterned farmland visible in satellite imagery.",

    "Forest": "Dense regions covered by trees and natural vegetation. Forests play a crucial role in biodiversity conservation, carbon storage, climate regulation, and maintaining ecological balance across large geographic areas.",

    "HerbaceousVegetation": "Areas dominated by grasses, shrubs, and low-lying vegetation rather than dense tree cover. These regions often serve as transitional ecosystems and are important for grazing and ecological diversity.",

    "Highway": "Major transportation corridors consisting of roads, highways, and associated infrastructure. These features are critical for connectivity between urban, industrial, and rural regions.",

    "Industrial": "Zones containing factories, warehouses, manufacturing facilities, and large-scale infrastructure. These areas are typically characterized by large buildings, paved surfaces, and organized layouts.",

    "Pasture": "Grass-covered land primarily used for livestock grazing and agricultural activities. Pasture regions often appear as open green landscapes with relatively uniform vegetation patterns.",

    "PermanentCrop": "Land dedicated to long-term cultivated crops such as orchards, vineyards, and plantations. These areas remain productive for multiple years without requiring annual replanting.",

    "Residential": "Urban and suburban settlement regions containing housing developments, roads, and community infrastructure. Residential areas typically show dense building patterns and organized layouts.",

    "River": "Natural flowing water bodies that transport water across landscapes. Rivers are critical components of ecosystems and often influence surrounding vegetation, agriculture, and settlements.",

    "SeaLake": "Large water bodies including lakes, reservoirs, coastal waters, and inland seas. These regions are characterized by extensive continuous water surfaces visible from satellite imagery."
}

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model(
        "geoscope_fixed.keras",
        compile=False,
        custom_objects={
            "preprocess_input": preprocess_input
        },
    )

model = load_my_model()

# ----------------------------
# HOME PAGE
# ----------------------------

st.markdown("""
<div class="hero">
    <div class="hero-title">
        GeoScope AI
    </div><div class="hero-subtitle">
        AI-Powered Satellite Land Use Classification
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div style="
text-align:center;
max-width:850px;
margin:auto;
line-height:1.6;
">

<p>
GeoScope AI is an AI-powered satellite image classification platform developed using Deep Learning and Computer Vision techniques.

The system is trained on the EuroSAT RGB Dataset, a widely used benchmark dataset for land-use and land-cover classification using satellite imagery. The model can automatically classify satellite images into 10 different land-use categories, including forests, rivers, highways, residential areas, industrial zones, crops, and water bodies.

To achieve high classification performance, the project leverages Transfer Learning with EfficientNetB0, enabling the model to learn powerful visual features from large-scale image datasets and adapt them for satellite image analysis.

The platform demonstrates how Artificial Intelligence can be applied to Earth Observation, Remote Sensing, Environmental Monitoring, Urban Planning, and Land-Use Analysis through automated image classification.
<b>Technologies Used</b><br>

TensorFlow • Keras • EfficientNetB0 • Transfer Learning • Streamlit • Computer Vision • Remote Sensing
</p>

</div>
""", unsafe_allow_html=True)

st.markdown("---")


# ----------------------------
# UPLOADER
# ----------------------------
left, center, right = st.columns([1,2,1])

with center:
 st.markdown(
 "<h3 style='text-align:center;'>Upload Satellite Image</h3>",
    unsafe_allow_html=True
)
 uploaded_file = st.file_uploader(
            "",
            type=["jpg", "jpeg", "png"]
        )

if uploaded_file is not None:
  image = Image.open(uploaded_file).convert("RGB")
  col1, col2 = st.columns([1, 1])

  with col1:

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        image = image.convert("RGB")
        img = image.resize((224, 224))

        img_array = np.array(img)

        img_array = np.expand_dims(
           img_array,
          axis=0
        )

        img_array = preprocess_input(
           img_array.astype(np.float32)
        )

        predictions = model.predict(img_array, verbose=0)

        pred_index = np.argmax(predictions)

        confidence = float(
           np.max(predictions) * 100
        )

        prediction = class_names[pred_index]

  with col2:

         st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True
        )

         st.markdown(
            f'<div class="prediction-text">{prediction}</div>',
            unsafe_allow_html=True
        )

         st.markdown(
            f'<div class="confidence-text">Confidence: {confidence:.2f}%</div>',
            unsafe_allow_html=True
        )

         st.progress(confidence / 100)

         st.markdown("### Land Use Description")

         st.write(
            descriptions[prediction]
        )
         st.markdown(
            '</div>',
            unsafe_allow_html=True
        )
if uploaded_file is not None:
# ====================================
# Prediction Analysis
# ====================================
         st.markdown("""
           <h2 style='text-align:center;
           margin-top:20px;
           margin-bottom:20px;'>
           Prediction Analysis
           </h2>
            """, unsafe_allow_html=True)

         top3 = np.argsort(
           predictions[0]
        )[-3:][::-1]

         chart_df = pd.DataFrame({
           "Class": [
            class_names[i]
            for i in top3
           ],
        "Confidence (%)": [
            float(predictions[0][i] * 100)
            for i in top3
           ]
        })

         fig = px.bar(
          chart_df,
          x="Confidence (%)",
          y="Class",
          orientation="h",
          text="Confidence (%)"
        )

         fig.update_layout(
           height=450,
           template="plotly_white",
           showlegend=False,
           margin=dict(l=20, r=20, t=20, b=20)
        )

         st.plotly_chart(
          fig,
          use_container_width=True
        )

         st.dataframe(
           chart_df,
           use_container_width=True
        )

         pdf_buffer = BytesIO()

         pdf = canvas.Canvas(pdf_buffer)

         pdf.setTitle("GeoScope AI Prediction Report")

         pdf.setFont("Helvetica-Bold", 18)
         pdf.drawString(50, 800, "GeoScope AI")

         pdf.setFont("Helvetica", 12)
         pdf.drawString(
          50,
          770,
          f"Generated: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"
        )

         pdf.drawString(
          50,
          730,
          f"Prediction: {prediction}"
        )

         pdf.drawString(
          50,
          710,
          f"Confidence: {confidence:.2f}%"
        )
         
         pdf.setFont("Helvetica-Bold", 14)
         pdf.drawString(50, 660, "Description")

         styles = getSampleStyleSheet()

         desc = Paragraph(
             descriptions[prediction],
             styles["BodyText"]
         )

         w, h = desc.wrap(500, 100)
         desc.drawOn(pdf, 50, 640 - h)
         pdf.save()

         pdf_buffer.seek(0)
         left, center, right = st.columns([3,2,3])

         with center:
            st.download_button(
               label="Download Prediction Report (PDF)",
               data=pdf_buffer,
               file_name="GeoScope_AI_Report.pdf",
               mime="application/pdf"
               )
         
st.markdown("---")

st.markdown("""
<hr>

<div style='text-align:center;
padding:20px;
color:#6b7280;'>
         
GeoScope AI © 2026
<br>
Designed & Developed by
<b>Khushi Soni</b>
<b><b>
            
<a href="https://github.com/Khushisoni702" target="_blank">
<img
src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"
width="34"
style="
background:white;
padding:6px;
border-radius:50%;
box-shadow:0 0 12px rgba(255,255,255,0.4);
">
</a>
</div>
""",
unsafe_allow_html=True)

