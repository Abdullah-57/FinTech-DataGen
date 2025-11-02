#!/usr/bin/env python3
"""
Quick test to verify the ensemble metrics fix
"""

import requests
import json
import time

def test_ensemble_metrics():
    """Test ensemble metrics specifically"""
    print("🎯 Testing Ensemble Metrics Fix")
    print("=" * 40)
    
    # Register and train ensemble
    print("1. Registering AAPL for ensemble...")
    payload = {"symbol": "AAPL", "model_types": ["ensemble"]}
    response = requests.post("http://localhost:5000/api/adaptive/register", json=payload, timeout=10)
    
    if response.status_code != 200:
        print("❌ Registration failed")
        return False
    
    print("2. Training ensemble model...")
    payload = {"symbol": "AAPL", "model_type": "ensemble"}
    response = requests.post("http://localhost:5000/api/adaptive/train", json=payload, timeout=120)
    
    if response.status_code == 200:
        data = response.json()
        if data.get('status') == 'success':
            metrics = data.get('metrics', {})
            print(f"✅ Ensemble training successful!")
            print(f"   RMSE: {metrics.get('rmse', 'N/A')}")
            print(f"   MAE: {metrics.get('mae', 'N/A')}")
            print(f"   MAPE: {metrics.get('mape', 'N/A')}%")
            
            # Check if metrics are non-zero
            has_real_metrics = any(metrics.get(m, 0) != 0.0 for m in ['rmse', 'mae', 'mape'])
            if has_real_metrics:
                print("   ✅ ENSEMBLE METRICS ARE NOW WORKING!")
                return True
            else:
                print("   ❌ Ensemble metrics are still zero")
                return False
        else:
            print(f"❌ Ensemble training failed: {data.get('message', 'Unknown error')}")
            return False
    else:
        print(f"❌ Ensemble training request failed: {response.status_code}")
        return False

def test_horizon_display():
    """Test if horizons are displaying correctly"""
    print("\n🔮 Testing Horizon Display")
    print("=" * 40)
    
    # Test different horizons
    for horizon in [1, 5, 10]:
        payload = {
            "symbol": "AAPL",
            "model_type": "ensemble",
            "horizon": horizon
        }
        
        response = requests.post("http://localhost:5000/api/adaptive/predict", json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                predictions = data.get('predictions', [])
                print(f"   {horizon}-day: got {len(predictions)} predictions")
                
                if len(predictions) == horizon:
                    print(f"   ✅ {horizon}-day horizon is correct")
                else:
                    print(f"   ❌ {horizon}-day horizon wrong: expected {horizon}, got {len(predictions)}")
            else:
                print(f"   ❌ {horizon}-day prediction failed")
        else:
            print(f"   ❌ {horizon}-day request failed")

def main():
    """Run quick tests"""
    print("🧪 Quick Fix Verification")
    print("=" * 50)
    
    # Test ensemble metrics
    ensemble_fixed = test_ensemble_metrics()
    
    # Test horizon display
    test_horizon_display()
    
    print("\n" + "=" * 50)
    if ensemble_fixed:
        print("✅ ENSEMBLE METRICS FIXED!")
        print("The ensemble model should now show proper RMSE, MAE, MAPE values.")
    else:
        print("❌ Ensemble metrics still need work.")
    
    print("\nNote: If horizons show correct lengths in the backend logs")
    print("but wrong in the frontend, the issue is in the React component.")

if __name__ == "__main__":
    main()