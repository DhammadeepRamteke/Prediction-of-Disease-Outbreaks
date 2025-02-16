import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Ensure the 'saved_models' directory exists
os.makedirs("saved_models", exist_ok=True)

def train_and_save_model(X, y, model_name):
    """
    Trains a RandomForest model, evaluates accuracy, and saves the trained model using joblib.
    """
    # Split the dataset into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy:.2f}")

    # Save the trained model
    model_path = f"saved_models/{model_name}.joblib"
    joblib.dump(model, model_path)
    print(f"✅ Model saved successfully at: {model_path}\n")

# ---------------- Load and Train Models ---------------- #

# 1️⃣ Diabetes Prediction Model
diabetes_df = pd.read_csv("data/cleaned_diabetes.csv")
X_diabetes = diabetes_df.drop(columns=["Outcome"])  # Features
y_diabetes = diabetes_df["Outcome"]  # Target variable
train_and_save_model(X_diabetes, y_diabetes, "diabetes_model")

# 2️⃣ Heart Disease Prediction Model
heart_df = pd.read_csv("data/cleaned_heart.csv")
X_heart = heart_df.drop(columns=["target"])
y_heart = heart_df["target"]
train_and_save_model(X_heart, y_heart, "heart_disease_model")

# 3️⃣ Parkinson’s Disease Prediction Model
parkinsons_df = pd.read_csv("data/cleaned_parkinsons.csv")
X_parkinsons = parkinsons_df.drop(columns=["status"])
y_parkinsons = parkinsons_df["status"]
train_and_save_model(X_parkinsons, y_parkinsons, "parkinsons_disease_model")

print("🎉 All models trained and saved successfully!")
