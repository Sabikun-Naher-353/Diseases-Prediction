# src/train_stroke_model.py
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = "D:/disease-predictor/data/stroke.csv"
MODEL_PATH = "D:/disease-predictor/models/stroke_model.pkl"
ENC_PATH = "D:/disease-predictor/models/stroke_label_encoders.pkl"
IMPUTER_PATH = "D:/disease-predictor/models/stroke_imputer.pkl"

# 0. Ensure dataset exists
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

# 1. Load
df = pd.read_csv(DATA_PATH)
print("Loaded dataset:", df.shape)
print("Original columns:", list(df.columns)[:20])

# 2. Normalize column names to lowercase and strip spaces (important!)
df.columns = [c.strip() for c in df.columns]
df.columns = [c.lower() for c in df.columns]
print("Normalized columns:", list(df.columns)[:20])

# 3. Replace common missing markers with actual NaN
df = df.replace({"n/a": pd.NA, "na": pd.NA, "": pd.NA})

# 4. Drop id if present
if "id" in df.columns:
    df = df.drop(columns=["id"])

# 5. Ensure target exists
if "stroke" not in df.columns:
    raise KeyError("Target column 'stroke' not found in dataset")

# 6. Columns we will use (based on your dataset)
# Adjust if your header uses different names - but keep them lowercase
feature_cols = [
    "gender", "age", "hypertension", "heart_disease", "ever_married",
    "work_type", "residence_type", "avg_glucose_level", "bmi", "smoking_status"
]

# verify columns exist
missing = [c for c in feature_cols if c not in df.columns]
if missing:
    raise KeyError(f"Missing expected columns in dataset: {missing}")

# 7. Impute numeric missing values (age, avg_glucose_level, bmi if present)
num_cols = [c for c in ["age", "avg_glucose_level", "bmi"] if c in df.columns]
imputer = SimpleImputer(strategy="median")
if num_cols:
    df[num_cols] = imputer.fit_transform(df[num_cols])
    print("Imputed numeric columns:", num_cols)

# 8. Encode categorical columns with LabelEncoder and save encoders
categorical_cols = [c for c in ["gender", "ever_married", "work_type", "residence_type", "smoking_status"] if c in df.columns]
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = df[col].astype(str)  # convert NaN to 'nan' string for stable mapping
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"Encoded {col}: classes -> {list(le.classes_)}")

# 9. Prepare X and y
X = df[feature_cols].copy()
y = df["stroke"].astype(int).copy()
print("Feature sample:\n", X.head(2))
print("Target counts:\n", y.value_counts())

# 10. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 11. Train model (balanced class weighting)
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# 12. Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# 13. Save model, encoders, imputer
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)
joblib.dump(label_encoders, ENC_PATH)
joblib.dump(imputer, IMPUTER_PATH)

print("Saved model ->", MODEL_PATH)
print("Saved encoders ->", ENC_PATH)
print("Saved imputer ->", IMPUTER_PATH)
