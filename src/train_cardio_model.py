# src/train_cardio_model.py
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = "D:/disease-predictor//disease-predictor/data/cardio.csv"
OUT_PIPE = "D:/disease-predictor//disease-predictor/models/cardio_pipeline.pkl"

# 0. Check dataset
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

# 1. Load and normalize column names
df = pd.read_csv(DATA_PATH)
df.columns = [c.strip().lower() for c in df.columns]  # lowercase for consistency
print("Loaded", df.shape)
print("Columns:", list(df.columns))

# 2. Replace common text missing markers
df = df.replace({"n/a": pd.NA, "na": pd.NA, "": pd.NA})

# 3. Drop id if present
if "id" in df.columns:
    df = df.drop(columns=["id"])

# 4. Define feature columns (based on your provided header)
feature_cols = ["age","gender","height","weight","ap_hi","ap_lo",
                "cholesterol","gluc","smoke","alco","active"]
for c in feature_cols:
    if c not in df.columns:
        raise KeyError(f"Expected column '{c}' not found in CSV")

# 5. Target column
if "cardio" not in df.columns:
    raise KeyError("Target column 'cardio' not found in CSV")
y = df["cardio"].astype(int)

# 6. X
X = df[feature_cols].copy()

# 7. Imputer + scaler + model pipeline
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"))
])

# 8. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 9. Fit pipeline
pipeline.fit(X_train, y_train)

# 10. Evaluate
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# 11. Save pipeline
os.makedirs(os.path.dirname(OUT_PIPE), exist_ok=True)
joblib.dump(pipeline, OUT_PIPE)
print("Saved pipeline ->", OUT_PIPE)
print("Model expects features in this order:", pipeline.named_steps['clf'].feature_names_in_ if hasattr(pipeline.named_steps['clf'], "feature_names_in_") else feature_cols)
