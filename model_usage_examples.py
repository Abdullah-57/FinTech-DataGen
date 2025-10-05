#!/usr/bin/env python3
"""
Examples of how to use uploaded models from Hugging Face
"""

import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download
import joblib
import tensorflow as tf

def download_and_use_traditional_models():
    """Example: Download and use traditional models"""
    
    # Download Moving Average model
    ma_model_path = hf_hub_download(
        repo_id="your_username/fintech-traditional-forecasters", 
        filename="moving_average_model.pkl"
    )
    
    # Download ARIMA model
    arima_model_path = hf_hub_download(
        repo_id="your_username/fintech-traditional-forecasters", 
        filename="arima_model.pkl"
    )
    
    # Load models
    ma_model = joblib.load(ma_model_path)
    arima_model = joblib.load(arima_model_path)
    
    # Create sample data
    sample_data = pd.Series([100, 101, 102, 103, 104, 105])
    
    # Make predictions
    ma_prediction = ma_model.predict(steps=3)
    arima_prediction = arima_model.predict(steps=3)
    
    print("Moving Average Predictions:", ma_prediction)
    print("ARIMA Predictions:", arima_prediction)

def download_and_use_neural_models():
    """Example: Download and use neural models"""
    
    # Download entire repository
    repo_path = snapshot_download(repo_id="your_username/fintech-neural-forecasters")
    
    # Load LSTM model
    lstm_model = tf.keras.models.load_model(f"{repo_path}/lstm_model")
    lstm_scaler = joblib.load(f"{repo_path}/lstm_model/scaler.pkl")
    
    # Load Transformer model
    transformer_model = tf.keras.models.load_model(f"{repo_path}/transformer_model")
    transformer_scaler = joblib.load(f"{repo_path}/transformer_model/scaler.pkl")
    
    print("✅ Neural models loaded successfully")
    print(f"LSTM Model Summary: {lstm_model.summary()}")
    print(f"Transformer Model Summary: {transformer_model.summary()}")

def download_and_use_ensemble():
    """Example: Download and use ensemble model"""
    
    # Download ensemble model
    ensemble_path = hf_hub_download(
        repo_id="your_username/fintech-ensemble-forecaster", 
        filename="ensemble_model.pkl"
    )
    
    # Load ensemble
    ensemble_model = joblib.load(ensemble_path)
    
    # Make predictions
    predictions = ensemble_model.predict(steps=5)
    print("Ensemble Predictions:", predictions)

if __name__ == "__main__":
    print("🔽 Downloading and testing models from Hugging Face...")
    
    try:
        download_and_use_traditional_models()
        download_and_use_neural_models()
        download_and_use_ensemble()
        print("✅ All models tested successfully!")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you've uploaded the models first and updated the username!")