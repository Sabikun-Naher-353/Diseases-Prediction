# src/cardio.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

PIPE_PATH = "D:/disease-predictor/disease-predictor/models/cardio_pipeline.pkl"

def run():
    st.title("🫀 Cardio Risk Prediction")
    st.write("Enter patient info to predict cardiovascular disease (cardio).")

    # Load pipeline
    try:
        pipeline = joblib.load(PIPE_PATH)
    except Exception as e:
        st.error(f"Error loading model pipeline: {e}")
        st.stop()

    # Input fields (match feature_cols used in training)
    age = st.number_input("Age (years)", min_value=0, max_value=120, value=50)
    gender = st.selectbox("Gender", [1,2], index=0, help="Use numeric codes consistent with your dataset")
    height = st.number_input("Height (cm)", min_value=30, max_value=250, value=168)
    weight = st.number_input("Weight (kg)", min_value=10, max_value=300, value=70)
    ap_hi = st.number_input("Systolic BP (ap_hi)", min_value=50, max_value=300, value=120)
    ap_lo = st.number_input("Diastolic BP (ap_lo)", min_value=30, max_value=200, value=80)
    cholesterol = st.selectbox("Cholesterol (1: normal, 2: above normal, 3: well above)", [1,2,3], index=0)
    gluc = st.selectbox("Glucose (1: normal, 2: above normal, 3: well above)", [1,2,3], index=0)
    smoke = st.selectbox("Smoker (0=No, 1=Yes)", [0,1], index=0)
    alco = st.selectbox("Alcohol use (0=No, 1=Yes)", [0,1], index=0)
    active = st.selectbox("Active (0=No, 1=Yes)", [0,1], index=1)

    if st.button("🔍 Predict Cardio"):
        try:
            data = pd.DataFrame([{
                "age": age,
                "gender": gender,
                "height": height,
                "weight": weight,
                "ap_hi": ap_hi,
                "ap_lo": ap_lo,
                "cholesterol": cholesterol,
                "gluc": gluc,
                "smoke": smoke,
                "alco": alco,
                "active": active
            }])

            # Ensure columns are in the same order as training pipeline expects
            # If pipeline contains a final estimator with feature_names_in_, use that:
            try:
                feature_order = list(pipeline.named_steps['clf'].feature_names_in_)
            except Exception:
                # fallback — use data.columns order
                feature_order = list(data.columns)

            data = data.reindex(columns=feature_order)

            pred = pipeline.predict(data)[0]
            prob = pipeline.predict_proba(data)[0] if hasattr(pipeline, "predict_proba") else None

            if pred == 1:
                st.error("⚠️ High risk of cardiovascular disease detected. Please consult a doctor.")
            else:
                st.success("✅ Low risk of cardiovascular disease.")

            if prob is not None:
                st.info(f"Confidence — class 0 / class 1: {prob[0]:.2%} / {prob[1]:.2%}")

        except Exception as e:
            st.error(f"Prediction error: {e}")
