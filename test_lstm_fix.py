#!/usr/bin/env python3
"""
Test LSTM fix for the TensorFlow error
"""

import requests
import json
import time

def test_lstm_training():
    """Test LSTM training specifically"""
    print("🧪 Testing LSTM Training Fix")
    print("=" * 40)
    
    base_url = "http://localhost:5000"
    
    # First register AAPL
    print("1. Registering AAPL...")
    register_payload = {
        "symbol": "AAPL",
        "model_types": ["sgd", "lstm", "ensemble"]
    }
    
    try:
        response = requests.post(f"{base_url}/api/adaptive/register", json=register_payload, timeout=10)
        if response.status_code == 200:
            print("✅ Registration successful")
        else:
            print(f"❌ Registration failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Registration error: {e}")
        return False
    
    # Test LSTM training specifically
    print("\n2. Testing LSTM training...")
    lstm_payload = {
        "symbol": "AAPL",
        "model_type": "lstm"
    }
    
    try:
        print("   Sending LSTM training request...")
        response = requests.post(f"{base_url}/api/adaptive/train", json=lstm_payload, timeout=60)
        
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ LSTM training successful!")
            print(f"   Status: {data.get('status')}")
            print(f"   Version: {data.get('version')}")
            if 'metrics' in data:
                metrics = data['metrics']
                print(f"   RMSE: {metrics.get('rmse', 'N/A')}")
                print(f"   MAE: {metrics.get('mae', 'N/A')}")
            return True
        else:
            print("❌ LSTM training failed")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('message', 'Unknown error')}")
            except:
                print(f"   Raw response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ LSTM training error: {e}")
        return False

def test_lstm_prediction():
    """Test LSTM prediction"""
    print("\n3. Testing LSTM prediction...")
    
    base_url = "http://localhost:5000"
    prediction_payload = {
        "symbol": "AAPL",
        "model_type": "lstm",
        "horizon": 5
    }
    
    try:
        response = requests.post(f"{base_url}/api/adaptive/predict", json=prediction_payload, timeout=30)
        
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                print("✅ LSTM prediction successful!")
                predictions = data.get('predictions', [])
                print(f"   Predictions: {predictions[:3]}..." if len(predictions) > 3 else f"   Predictions: {predictions}")
                return True
            else:
                print(f"❌ LSTM prediction failed: {data.get('message')}")
                return False
        else:
            print("❌ LSTM prediction request failed")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('message', 'Unknown error')}")
            except:
                print(f"   Raw response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ LSTM prediction error: {e}")
        return False

def main():
    """Run LSTM tests"""
    print("🔧 LSTM Fix Verification")
    print("=" * 50)
    
    # Check if backend is running
    try:
        response = requests.get("http://localhost:5000/api/health", timeout=5)
        if response.status_code != 200:
            print("❌ Backend is not running!")
            print("Please start the backend with: cd backend && python app.py")
            return
    except:
        print("❌ Cannot connect to backend!")
        print("Please start the backend with: cd backend && python app.py")
        return
    
    print("✅ Backend is running")
    
    # Test LSTM training
    training_success = test_lstm_training()
    
    # Test LSTM prediction (only if training succeeded)
    prediction_success = False
    if training_success:
        prediction_success = test_lstm_prediction()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"   LSTM Training: {'✅ PASS' if training_success else '❌ FAIL'}")
    print(f"   LSTM Prediction: {'✅ PASS' if prediction_success else '❌ FAIL'}")
    
    if training_success and prediction_success:
        print("\n🎉 LSTM fix successful! The TensorFlow error has been resolved.")
        print("\nYou can now:")
        print("1. Use the LSTM model in the adaptive learning showcase")
        print("2. Train LSTM models without TensorFlow errors")
        print("3. Make predictions with trained LSTM models")
    else:
        print("\n⚠️  Some issues remain. Check the error messages above.")

if __name__ == "__main__":
    main()