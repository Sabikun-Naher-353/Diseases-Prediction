# src/train_sleep_model.py
"""
Train a sleep-disorder classifier pipeline and save the pipeline + label encoder.

Saves:
 - D:/disease-predictor/models/sleep_pipeline.pkl
 - D:/disease-predictor/models/sleep_label_encoder.pkl
"""
import os
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Paths (update if your folder structure differs)
DATA_PATH = "D:/disease-predictor/data/sleep.csv"
OUT_PIPE = "D:/disease-predictor/models/sleep_pipeline.pkl"
OUT_LE = "D:/disease-predictor/models/sleep_label_encoder.pkl"

# -------------------------
# 0. Load dataset
# -------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print("Loaded:", df.shape)
print("Original columns:", list(df.columns))

# -------------------------
# 1. Normalize column names
# -------------------------
# Convert to snake_case-ish lowercase names to avoid mismatches
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
print("Normalized columns:", list(df.columns))

# -------------------------
# 2. Parse blood pressure -> sys_bp, dia_bp
# -------------------------
def parse_bp(x):
    try:
        if pd.isna(x):
            return (pd.NA, pd.NA)
        s = str(x).strip()
        if "/" in s:
            a, b = s.split("/", 1)
            return (float(a), float(b))
        # if single number, treat as systolic
        return (float(s), pd.NA)
    except Exception:
        return (pd.NA, pd.NA)

if "blood_pressure" in df.columns:
    df[["sys_bp", "dia_bp"]] = df["blood_pressure"].apply(lambda x: pd.Series(parse_bp(x)))
    print("Parsed blood_pressure into sys_bp & dia_bp")

# -------------------------
# 3. Drop identifier columns (if any)
# -------------------------
if "person_id" in df.columns:
    df = df.drop(columns=["person_id"])
    print("Dropped person_id")

# -------------------------
# 4. Target check & prepare
# -------------------------
if "sleep_disorder" not in df.columns:
    raise KeyError("Target column 'sleep_disorder' not found in the dataset")

y_raw = df["sleep_disorder"].astype(str).str.strip()
print("Unique target values:", y_raw.unique())

# -------------------------
# 5. Feature selection
# -------------------------
# Numeric features we expect (only include if present)
numeric_candidates = [
    "age", "sleep_duration", "quality_of_sleep",
    "physical_activity_level", "stress_level",
    "heart_rate", "daily_steps", "sys_bp", "dia_bp"
]
num_features = [c for c in numeric_candidates if c in df.columns]

# Categorical features
cat_candidates = ["gender", "occupation", "bmi_category"]
cat_features = [c for c in cat_candidates if c in df.columns]

print("Numeric features:", num_features)
print("Categorical features:", cat_features)

if len(num_features) + len(cat_features) == 0:
    raise ValueError("No features detected. Check CSV headers after normalization.")

X = df[num_features + cat_features].copy()

# -------------------------
# 6. Preprocessing pipelines
# -------------------------
# Numeric pipeline: median impute -> scale
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical pipeline: fill missing -> one-hot
# Note: use sparse_output=False for sklearn>=1.4 compatibility
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preproc = ColumnTransformer(
    transformers=[
        ("num", num_pipe, num_features),
        ("cat", cat_pipe, cat_features)
    ],
    remainder="drop"
)

# -------------------------
# 7. Encode target labels
# -------------------------
le = LabelEncoder()
y = le.fit_transform(y_raw)
print("Target classes (label encoder):", list(le.classes_))

# -------------------------
# 8. Model pipeline
# -------------------------
clf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"  # helps with class imbalance
)

pipeline = Pipeline([
    ("preproc", preproc),
    ("clf", clf)
])

# -------------------------
# 9. Train / test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# -------------------------
# 10. Fit
# -------------------------
pipeline.fit(X_train, y_train)
print("Training complete.")

# -------------------------
# 11. Evaluate
# -------------------------
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# -------------------------
# 12. Save pipeline + label encoder
# -------------------------
os.makedirs(os.path.dirname(OUT_PIPE), exist_ok=True)
joblib.dump(pipeline, OUT_PIPE)
joblib.dump(le, OUT_LE)
print("Saved pipeline ->", OUT_PIPE)
print("Saved label encoder ->", OUT_LE)
