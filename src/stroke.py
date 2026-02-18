# src/stroke.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

MODEL_PATH = "D:/disease-predictor/models/stroke_model.pkl"
ENC_PATH = "D:/disease-predictor/models/stroke_label_encoders.pkl"
IMPUTER_PATH = "D:/disease-predictor/models/stroke_imputer.pkl"

def run():
    st.title("Stroke Risk Prediction")
    st.write("Enter the patient details below:")

    # Load model, encoders and imputer
    try:
        model = joblib.load(MODEL_PATH)
        label_encoders = joblib.load(ENC_PATH)
        imputer = joblib.load(IMPUTER_PATH)
    except Exception as e:
        st.error(f"Error loading model/encoders/imputer: {e}")
        st.stop()

    # Build input options from saved encoders (so they match training)
    gender_opts = list(label_encoders.get("gender").classes_) if "gender" in label_encoders else ["Male","Female"]
    ever_married_opts = list(label_encoders.get("ever_married").classes_) if "ever_married" in label_encoders else ["Yes","No"]
    work_type_opts = list(label_encoders.get("work_type").classes_) if "work_type" in label_encoders else ["Private","Self-employed","Govt_job","children","Never_worked"]
    residence_opts = list(label_encoders.get("residence_type").classes_) if "residence_type" in label_encoders else list(label_encoders.get("Residence_type").classes_) if "Residence_type" in label_encoders else ["Urban","Rural"]
    smoking_opts = list(label_encoders.get("smoking_status").classes_) if "smoking_status" in label_encoders else ["never smoked","formerly smoked","smokes","Unknown"]

    # Input widgets
    gender = st.selectbox("Gender", gender_opts, index=gender_opts.index(gender_opts[0]))
    age = st.number_input("Age", min_value=0, max_value=120, value=65)
    hypertension = st.selectbox("Hypertension (0=No,1=Yes)", [0,1], index=0)
    heart_disease = st.selectbox("Heart disease (0=No,1=Yes)", [0,1], index=0)
    ever_married = st.selectbox("Ever married", ever_married_opts, index=0)
    work_type = st.selectbox("Work type", work_type_opts, index=work_type_opts.index(work_type_opts[0]))
    residence_type = st.selectbox("Residence type", residence_opts, index=0)
    avg_glucose_level = st.number_input("Average glucose level", min_value=0.0, max_value=1000.0, value=120.0)
    bmi = st.number_input("BMI", min_value=5.0, max_value=80.0, value=25.0)
    smoking_status = st.selectbox("Smoking status", smoking_opts, index=0)

    if st.button("🔍 Predict Stroke Risk"):
        try:
            # Build dict with lowercase keys to match training
            data = {
                "gender": gender,
                "age": age,
                "hypertension": int(hypertension),
                "heart_disease": int(heart_disease),
                "ever_married": ever_married,
                "work_type": work_type,
                "residence_type": residence_type,
                "avg_glucose_level": avg_glucose_level,
                "bmi": bmi,
                "smoking_status": smoking_status
            }

            # Encode categorical features using saved encoders
            for col, le in label_encoders.items():
                # some models saved 'Residence_type' instead of 'residence_type' - handle both
                key = col.lower()
                if key in data:
                    try:
                        data[key] = int(le.transform([str(data[key])])[0])
                    except Exception:
                        # fallback to first class encoding if unseen
                        data[key] = int(le.transform([le.classes_[0]])[0])
                else:
                    # handle weird encoder keys like 'Residence_type'
                    if col in data:
                        try:
                            data[col] = int(le.transform([str(data[col])])[0])
                        except Exception:
                            data[col] = int(le.transform([le.classes_[0]])[0])

            # Prepare DataFrame with same column order as model expects
            feature_names = list(model.feature_names_in_)
            input_df = pd.DataFrame([data])[feature_names]

            # Apply imputer (same that was used in training)
            input_df[input_df.columns] = imputer.transform(input_df[input_df.columns])

            # Predict
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0] if hasattr(model, "predict_proba") else None

            if pred == 1:
                st.error("⚠️ High stroke risk detected. Please consult a doctor.")
            else:
                st.success("✅ Low stroke risk detected.")

            if prob is not None:
                st.info(f"Model confidence (class 0 / class 1): {prob[0]:.2%} / {prob[1]:.2%}")

        except Exception as e:
            st.error(f"Prediction error: {e}")

if __name__ == "__main__":
    run()
