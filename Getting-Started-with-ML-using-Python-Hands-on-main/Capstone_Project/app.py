import streamlit as st
import numpy as np
import joblib

model=joblib.load("breast_cancer_model.pkl")
scaler=joblib.load("scaler.pkl")

st.set_page_config(page_title="Breast Cancer Detection",
                   page_icon="🩺",
                   layout="wide")
st.title("🩺 Breast Cancer Classification System")
st.write("""Predict whether a tumor is Benign or Malignant""")
feature_names=['radius_mean','texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se',
    'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
    'smoothness_worst', 'compactness_worst', 'concavity_worst',
    'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']
inputs=[]
col1, col2 = st.columns(2)
for i,feature in enumerate(feature_names):
    if i<15:
        value=col1.number_input(feature,value =0.0)
    else:
        value = col2.number_input(feature,value=0.0)
    inputs.append(value)
if st.button("Predict"):
    input_array = np.array(inputs).reshape(1,-1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    st.subheader("Prediction Result")
    if prediction ==1:
        st.error("Malignant Tumor Detected")
    else:
        st.success("Benign Tumor Detected")
st.warning("This application is intended for education and research purposes only and should not be used as a substitute for professional medical advice")
    