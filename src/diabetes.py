import streamlit as st
import pandas as pd
import numpy as np
import joblib

def run():
    st.title("🩺 Diabetes Risk Prediction")
    
    try:
        model = joblib.load('D:/disease-predictor/disease-predictor/models/real_model.pkl')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
    bp = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
    skin = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=79)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
    age = st.number_input("Age", min_value=1, max_value=120, value=33)
    if st.button("🔍 Predict"):
        input_data = pd.DataFrame([{
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": bp,
            "SkinThickness": skin,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age
        }])
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.error("⚠️ High risk of diabetes detected! Please consult a doctor.")
        else:
            st.success("✅ No diabetes risk detected. You seem healthy!")
