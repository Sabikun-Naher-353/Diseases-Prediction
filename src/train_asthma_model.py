import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import os

DATA_PATH = "data/asthma.csv"
MODEL_PATH = "models/asthma_model.pkl"
ENC_PATH = "models/asthma_label_encoders.pkl"

os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)
print("Loaded:", df.shape)
print(df.head())

# Drop ID
df = df.drop(columns=["Patient_ID"])

# Encode categorical columns
encoders = {}
for col in ["Gender", "Smoking_Status", "Medication"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Encode target
target_enc = LabelEncoder()
df["Asthma_Diagnosis"] = target_enc.fit_transform(df["Asthma_Diagnosis"])
encoders["Asthma_Diagnosis"] = target_enc

# Features and target
X = df.drop(columns=["Asthma_Diagnosis"])
y = df["Asthma_Diagnosis"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model and encoders
joblib.dump(model, MODEL_PATH)
joblib.dump(encoders, ENC_PATH)
print("\n✅ Model and encoders saved successfully.")
