"""
Backwards-compatible model training script with modern scikit-learn models.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib, pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "data.csv"
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)


def get_clean_data():
    data = pd.read_csv(DATA_PATH)
    if "Unnamed: 32" in data.columns:
        data = data.drop(["Unnamed: 32"], axis=1)
    if "id" in data.columns:
        data = data.drop(["id"], axis=1)
    data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})
    return data


def create_model(data):
    X = data.drop(["diagnosis"], axis=1)
    y = data["diagnosis"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    model = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification report:
", classification_report(y_test, y_pred))
    return model, scaler


def main():
    data = get_clean_data()
    model, scaler = create_model(data)
    with open(MODEL_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f, protocol=4)
    with open(MODEL_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f, protocol=4)
    print("Models saved successfully to model/!")


if __name__ == "__main__":
    main()
