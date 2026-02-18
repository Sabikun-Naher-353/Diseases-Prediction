# src/debug_stroke.py
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = "D:/disease-predictor/models/stroke_model.pkl"
ENC_PATH = "D:/disease-predictor/models/stroke_label_encoders.pkl"
IMPUTER_PATH = "D:/disease-predictor/models/stroke_imputer.pkl"
DATA_PATH = "D:/disease-predictor/data/stroke.csv"

print("1) Load model, encoders, imputer")
model = joblib.load(MODEL_PATH)
print(" model type:", type(model))
print(" classes_:", getattr(model, "classes_", "no classes_ attribute"))
print(" feature_names_in_:", getattr(model, "feature_names_in_", None))

encoders = None
imputer = None
try:
    encoders = joblib.load(ENC_PATH)
    print(" Loaded encoders keys:", list(encoders.keys()))
except Exception as e:
    print(" No encoders loaded:", e)

try:
    imputer = joblib.load(IMPUTER_PATH)
    print(" Loaded imputer:", type(imputer))
except Exception as e:
    print(" No imputer loaded:", e)

print("\n2) Load dataset & inspect label distribution")
df = pd.read_csv(DATA_PATH)
print(" dataset shape:", df.shape)
if "stroke" in df.columns:
    print(" stroke value counts:\n", df["stroke"].value_counts(dropna=False))
else:
    print(" WARNING: 'stroke' column not found in CSV")

print("\n3) Show a few rows (head) and sample high-risk rows")
print(df.head(8))
high = df[df["stroke"]==1]
print(" Count high-risk(rows==1):", len(high))
print(" Example high-risk rows (first 5):")
print(high.head(5))

print("\n4) Check training encoding assumptions")
if encoders:
    for c, le in encoders.items():
        print(f" Encoder for {c} classes -> {list(le.classes_)}")

print("\n5) Pick an example high-risk row and run full transform -> predict")
if len(high) == 0:
    print(" No high-risk rows found in CSV to test.")
else:
    r = high.iloc[0].to_dict()
    print(" Raw row:", r)

    # Build data dictionary using model.feature_names_in_ if available, otherwise known order
    feature_names = list(getattr(model, "feature_names_in_", [
        "gender","age","hypertension","heart_disease","ever_married",
        "work_type","Residence_type","avg_glucose_level","bmi","smoking_status"
    ]))
    print(" Using feature_names:", feature_names)

    # If encoders present, transform categorical fields; else try simple maps
    def transform_value(col, val):
        if encoders and col in encoders:
            try:
                return int(encoders[col].transform([str(val)])[0])
            except Exception as e:
                print(f"  Encoder transform failed for {col} val={val} -> {e}; using fallback class[0]")
                return int(encoders[col].transform([encoders[col].classes_[0]])[0])
        # fallback heuristics (common patterns)
        if col == "gender":
            return 1 if str(val).lower().startswith("m") else 0
        if col == "Residence_type":
            return 1 if str(val).lower().startswith("u") else 0
        if col == "smoking_status":
            v = str(val).lower()
            if "never" in v: return 0
            if "former" in v: return 1
            if "smoke" in v: return 2
            return 3
        # numeric columns return as-is
        try:
            return float(val)
        except:
            return 0.0

    transformed = {}
    for col in feature_names:
        raw_val = r.get(col, None)
        # If column not in CSV and model expects it, try alternate casing:
        if raw_val is None:
            # try lowercase/uppercase variants
            for alt in [col.lower(), col.upper(), col.capitalize()]:
                if alt in r:
                    raw_val = r[alt]; break
        transformed[col] = transform_value(col, raw_val)
    print(" Transformed input:", transformed)

    X_test = pd.DataFrame([transformed], columns=feature_names)
    print(" Input DataFrame:\n", X_test)

    # apply imputer if model was trained with imputer
    if imputer is not None:
        try:
            X_test[X_test.columns] = imputer.transform(X_test[X_test.columns])
            print(" After imputer:", X_test)
        except Exception as e:
            print(" Imputer transform failed:", e)

    # Final predict + proba
    try:
        pred = model.predict(X_test)
        print(" model.predict ->", pred)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_test)
            print(" model.predict_proba ->", prob)
    except Exception as e:
        print(" Prediction failed:", e)

print("\n6) Quick sanity checks if model predicts only 0 always")
# check performance on full test split if pipeline exists inside models (if you saved pipeline, run this)
try:
    if hasattr(model, "predict") and (df is not None) and ("stroke" in df.columns):
        # try to evaluate on some rows (use small sample)
        # Build X from df using the same feature names
        sample_df = df.copy()
        # ensure numeric columns handled
        feats = feature_names
        missing_cols = [c for c in feats if c not in sample_df.columns]
        if missing_cols:
            print(" Warning: CSV missing these model-features:", missing_cols)
        else:
            X_all = sample_df[feats].copy()
            # if encoders are present, transform columns
            if encoders:
                for c in encoders:
                    if c in X_all.columns:
                        X_all[c] = encoders[c].transform(X_all[c].astype(str))
            # impute if available
            if imputer is not None:
                X_all[X_all.columns] = imputer.transform(X_all)
            preds = model.predict(X_all)
            print(" Predictions on CSV sample (value counts):", pd.Series(preds).value_counts())
            print(" Confusion matrix vs true:\n", confusion_matrix(sample_df["stroke"], preds))
            print(" Classification report:\n", classification_report(sample_df["stroke"], preds))
except Exception as e:
    print(" Could not run full CSV eval:", e)

print("\n7) Recommendations (printed after diagnostics).")
