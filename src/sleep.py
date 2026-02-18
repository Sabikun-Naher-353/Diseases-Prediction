# src/sleep.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

PIPE_PATH = "D:/disease-predictor/disease-predictor/models/sleep_pipeline.pkl"
LE_PATH = "D:/disease-predictor/disease-predictor/models/sleep_label_encoder.pkl"

def run():
    st.title("🛌 Sleep Disorder Prediction")
    st.write("Enter user details to predict likely sleep disorder.")

    # Load pipeline and label encoder
    try:
        pipeline = joblib.load(PIPE_PATH)
        le = joblib.load(LE_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    # — Inputs (match features used in training)
    gender = st.selectbox("Gender", ["Male","Female","Other"])
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    occupation = st.text_input("Occupation", value="Software Engineer")
    sleep_duration = st.number_input("Sleep duration (hours)", min_value=0.0, max_value=24.0, value=6.5, step=0.1)
    quality_of_sleep = st.number_input("Quality of sleep (1-10)", min_value=0, max_value=10, value=6)
    physical_activity_level = st.number_input("Physical activity level (minutes/day or score)", min_value=0, max_value=1000, value=40)
    stress_level = st.number_input("Stress level (1-10)", min_value=0, max_value=10, value=5)
    bmi_category = st.selectbox("BMI Category", ["Underweight","Normal","Overweight","Obese","missing"])
    sys_bp = st.number_input("Systolic BP (sys_bp)", min_value=0, max_value=300, value=120)
    dia_bp = st.number_input("Diastolic BP (dia_bp)", min_value=0, max_value=200, value=80)
    heart_rate = st.number_input("Heart rate (bpm)", min_value=0, max_value=300, value=75)
    daily_steps = st.number_input("Daily steps", min_value=0, max_value=100000, value=4000)

    if st.button("🔍 Predict Sleep Disorder"):
        # Build DataFrame using the same feature names expected in training
        data = {
            "age": age,
            "sleep_duration": sleep_duration,
            "quality_of_sleep": quality_of_sleep,
            "physical_activity_level": physical_activity_level,
            "stress_level": stress_level,
            "heart_rate": heart_rate,
            "daily_steps": daily_steps,
            "sys_bp": sys_bp,
            "dia_bp": dia_bp,
            "gender": gender,
            "occupation": occupation,
            "bmi_category": bmi_category
        }

        input_df = pd.DataFrame([data])

        try:
            pred = pipeline.predict(input_df)[0]
            proba = pipeline.predict_proba(input_df)[0] if hasattr(pipeline, "predict_proba") else None
            label = le.inverse_transform([pred])[0]

            st.write("### Prediction")
            st.write(f"**Predicted sleep disorder:** {label}")
            if proba is not None:
                # show the probability for each class
                classes = le.classes_
                probs = {cls: f"{prob:.2%}" for cls, prob in zip(classes, proba)}
                st.write("**Probabilities:**")
                st.json(probs)
        except Exception as e:
            st.error(f"Prediction error: {e}")

if __name__ == "__main__":
    run()
