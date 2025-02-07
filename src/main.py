import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import joblib
import os

def preprocess_data(file_path):
    print("Loading dataset...")
    data = pd.read_csv(file_path)
    print(f"Dataset loaded with shape: {data.shape}")
    
    print("Splitting features and target...")
    X = data.drop(columns=["Class"])
    y = data["Class"]
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Splitting into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    print(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, X_test, y_test, model_path):
    print("Training Isolation Forest model...")
    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(X_train)
    print("Model training complete.")
    
    print("Making predictions...")
    predictions = model.predict(X_test)
    predictions = [1 if pred == -1 else 0 for pred in predictions]
    
    print("Evaluation metrics:")
    print(classification_report(y_test, predictions))
    
    print(f"Saving model to {model_path}...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print("Model saved successfully.")

def evaluate_model(model_path, X_test, y_test):
    print("Loading model...")
    model = joblib.load(model_path)
    print("Model loaded successfully.")
    
    print("Making predictions...")
    predictions = model.predict(X_test)
    predictions = [1 if pred == -1 else 0 for pred in predictions]
    
    print("Classification Report:")
    print(classification_report(y_test, predictions))

def main():
    file_path = "data/raw/creditcard.csv"
    model_path = "models/isolation_forest_model.pkl"
    
    X_train, X_test, y_train, y_test = preprocess_data(file_path)
    train_model(X_train, X_test, y_test, model_path)
    evaluate_model(model_path, X_test, y_test)

if __name__ == "__main__":
    main()
