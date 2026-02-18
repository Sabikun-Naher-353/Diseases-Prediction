import streamlit as st
import pandas as pd
import joblib
import os

MODEL_PATH = "models/asthma_model.pkl"
ENC_PATH = "models/asthma_label_encoders.pkl"

def run():
    st.title("🌬️ Asthma Risk Prediction")

    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENC_PATH):
        st.error("Model not trained yet. Please run train_asthma_model.py first.")
        return

    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENC_PATH)

    gender = st.selectbox("Gender", encoders["Gender"].classes_)
    smoking_status = st.selectbox("Smoking Status", encoders["Smoking_Status"].classes_)
    
    medication = st.selectbox("Medication", encoders["Medication"].classes_)
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    peak_flow = st.number_input("Peak Flow", min_value=100, max_value=500, value=250)

    if st.button("Predict Risk"):
        data = pd.DataFrame([{
            "Age": age,
            "Gender": encoders["Gender"].transform([gender])[0],
            "Smoking_Status": encoders["Smoking_Status"].transform([smoking_status])[0],
            "Medication": encoders["Medication"].transform([medication])[0],
            "Peak_Flow": peak_flow
        }])

        prediction = model.predict(data)[0]
        result = encoders["Asthma_Diagnosis"].inverse_transform([prediction])[0]

        if result == "Yes":
            st.error("🔴 High Risk of Asthma")
        else:
            st.success("🟢 Low Risk of Asthma")

if __name__ == "__main__":
    run()
