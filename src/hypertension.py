import streamlit as st
import pandas as pd
import joblib
import os

PIPE_PATH = "D:/disease-predictor/disease-predictor/models/hypertension_pipeline.pkl"

def run():
    st.title("🫀 Hypertension Risk Prediction")
    st.write("Enter patient data (features must match training).")

    try:
        pipeline = joblib.load(PIPE_PATH)
    except Exception as e:
        st.error(f"Error loading model pipeline: {e}")
        st.stop()
    gender = st.selectbox("Gender (0=Female, 1=Male)", [0,1], index=0)
    age = st.number_input("Age", min_value=0, max_value=120, value=45)
    currentSmoker = st.selectbox("Current Smoker (0=No, 1=Yes)", [0,1], index=0)
    cigsPerDay = st.number_input("Cigarettes per day", min_value=0, max_value=200, value=0)
    BPMeds = st.selectbox("On BP medication (BPMeds) (0=No, 1=Yes)", [0,1], index=0)
    diabetes = st.selectbox("Diabetes (0=No, 1=Yes)", [0,1], index=0)
    totChol = st.number_input("Total Cholesterol (mg/dL)", min_value=0, max_value=1000, value=190)
    sysBP = st.number_input("Systolic BP (sysBP)", min_value=50, max_value=300, value=120)
    diaBP = st.number_input("Diastolic BP (diaBP)", min_value=30, max_value=200, value=80)
    BMI = st.number_input("BMI", min_value=10.0, max_value=60.0, value=26.0)
    heartRate = st.number_input("Heart Rate", min_value=30, max_value=200, value=75)
    glucose = st.number_input("Glucose", min_value=0, max_value=500, value=90)

    if st.button("🔍 Predict Hypertension Risk"):
        input_df = pd.DataFrame([{
            "gender": gender,
            "age": age,
            "currentSmoker": currentSmoker,
            "cigsPerDay": cigsPerDay,
            "BPMeds": BPMeds,
            "diabetes": diabetes,
            "totChol": totChol,
            "sysBP": sysBP,
            "diaBP": diaBP,
            "BMI": BMI,
            "heartRate": heartRate,
            "glucose": glucose
        }])

        try:
            pred = pipeline.predict(input_df)
            prob = pipeline.predict_proba(input_df) if hasattr(pipeline, "predict_proba") else None

            if pred[0] == 1:
                st.error("⚠️ High risk of hypertension detected. Please consult a doctor.")
            else:
                st.success("✅ Low risk of hypertension detected.")

            if prob is not None:
                st.info(f"Model confidence: {prob.max():.2%}")

        except Exception as e:
            st.error(f"Prediction error: {e}")
