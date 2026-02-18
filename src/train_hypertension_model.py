import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = "D:/disease-predictor/data/hypertension.csv"
OUT_PATH = "D:/disease-predictor/models/hypertension_pipeline.pkl"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print("Loaded:", df.shape)
print("Columns:", list(df.columns))

df["Risk"] = df["Risk"].apply(lambda x: int(x))

feature_cols = ["gender","age","currentSmoker","cigsPerDay","BPMeds",
                "diabetes","totChol","sysBP","diaBP","BMI","heartRate","glucose"]
X = df[feature_cols].copy()
y = df["Risk"].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
print("Classification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
joblib.dump(pipeline, OUT_PATH)
print(f"Saved pipeline -> {OUT_PATH}")
