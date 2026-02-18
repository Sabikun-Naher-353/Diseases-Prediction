import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# 1️⃣ Load dataset
DATA_PATH = r"D:\disease-predictor\data\diabetes.csv"

data = pd.read_csv(DATA_PATH)

TARGET_COL = "Outcome"  # ⚠️ replace with your actual target column name
X = data.drop(columns=[TARGET_COL])
y = data[TARGET_COL]

# 2️⃣ Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3️⃣ Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 4️⃣ Evaluate model
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Test Accuracy: {acc:.4f}")

# 5️⃣ Save trained model
os.makedirs("../models", exist_ok=True)
joblib.dump(clf, "D:/disease-predictor/models/dummy_model.pkl")

print("💾 Model saved to ../models/dummy_model.pkl")

