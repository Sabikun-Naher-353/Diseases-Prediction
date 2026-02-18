import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

data = pd.read_csv("data/diabetes.csv")

print("✅ Dataset loaded successfully.")
print("Columns:", list(data.columns))
print("Total rows:", len(data))

X = data.drop('Outcome', axis=1)
y = data['Outcome']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200, max_depth=None, random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {acc:.4f}")
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

joblib.dump(model, "models/real_model.pkl")
print("\n💾 Model saved to models/real_model.pkl")
