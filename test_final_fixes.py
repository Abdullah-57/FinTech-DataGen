#!/usr/bin/env python3
"""
Final test script to verify ensemble metrics and horizon fixes
"""

import requests
import json
import time

def test_backend_health():
    """Test if backend is running"""
    try:
        response = requests.get("http://localhost:5000/api/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is running and healthy")
            return True
        else:
            print(f"❌ Backend health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend. Is it running on port 5000?")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_ensemble_metrics():
    """Test ensemble model metrics specifically"""
    print("\n🎯 Testing Ensemble Metrics Fix")
    print("=" * 50)
    
    # Register symbol
    payload = {"symbol": "AAPL", "model_types": ["ensemble"]}
    response = requests.post("http://localhost:5000/api/adaptive/register", json=payload, timeout=10)
    
    if response.status_code != 200:
        print("❌ Registration failed")
        return False
    
    # Train ensemble model
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

def test_horizon_predictions():
    """Test horizon predictions for all models"""
    print("\n🔮 Testing Horizon Predictions Fix")
    print("=" * 50)
    
    # Register all models
    payload = {"symbol": "AAPL", "model_types": ["sgd", "lstm", "ensemble"]}
    response = requests.post("http://localhost:5000/api/adaptive/register", json=payload, timeout=10)
    
    if response.status_code != 200:
        print("❌ Registration failed")
        return False
    
    # Train all models
    training_success = {}
    for model_type in ["sgd", "lstm", "ensemble"]:
        payload = {"symbol": "AAPL", "model_type": model_type}
        response = requests.post("http://localhost:5000/api/adaptive/train", json=payload, timeout=120)
        
        if response.status_code == 200 and response.json().get('status') == 'success':
            training_success[model_type] = True
            print(f"✅ {model_type.upper()} trained successfully")
        else:
            training_success[model_type] = False
            print(f"❌ {model_type.upper()} training failed")
        
        time.sleep(1)  # Brief pause
    
    # Test predictions for each horizon
    horizon_results = {}
    for model_type in ["sgd", "lstm", "ensemble"]:
        if not training_success.get(model_type, False):
            print(f"⏭️  Skipping {model_type.upper()} predictions (training failed)")
            continue
            
        print(f"\n📈 Testing {model_type.upper()} horizon predictions...")
        horizon_results[model_type] = {}
        
        for horizon in [1, 5, 10]:
            payload = {
                "symbol": "AAPL",
                "model_type": model_type,
                "horizon": horizon
            }
            
            response = requests.post("http://localhost:5000/api/adaptive/predict", json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    predictions = data.get('predictions', [])
                    actual_horizon = len(predictions)
                    
                    print(f"   {horizon}-day: requested={horizon}, got={actual_horizon}")
                    
                    if actual_horizon == horizon:
                        print(f"   ✅ CORRECT! {model_type.upper()} {horizon}-day prediction works")
                        horizon_results[model_type][horizon] = True
                    else:
                        print(f"   ❌ WRONG! Expected {horizon}, got {actual_horizon}")
                        horizon_results[model_type][horizon] = False
                else:
                    print(f"   ❌ {horizon}-day prediction failed: {data.get('message', 'Unknown error')}")
                    horizon_results[model_type][horizon] = False
            else:
                print(f"   ❌ {horizon}-day prediction request failed: {response.status_code}")
                horizon_results[model_type][horizon] = False
    
    return horizon_results

def main():
    """Run all tests"""
    print("🧪 Final Fix Verification Test")
    print("=" * 60)
    
    # Check backend health
    if not test_backend_health():
        print("\n❌ Backend is not running. Please start it with:")
        print("   cd backend")
        print("   python app.py")
        return
    
    # Test ensemble metrics
    ensemble_metrics_fixed = test_ensemble_metrics()
    
    # Test horizon predictions
    horizon_results = test_horizon_predictions()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 FINAL TEST RESULTS:")
    
    print(f"\n🎯 Ensemble Metrics Fix:")
    if ensemble_metrics_fixed:
        print("   ✅ FIXED! Ensemble now shows proper RMSE, MAE, MAPE values")
    else:
        print("   ❌ STILL BROKEN! Ensemble metrics are still zero")
    
    print(f"\n🔮 Horizon Predictions Fix:")
    all_horizons_correct = True
    
    for model_type, horizons in horizon_results.items():
        print(f"   {model_type.upper()}:")
        for horizon, success in horizons.items():
            status = "✅ FIXED" if success else "❌ BROKEN"
            print(f"     {horizon}-day: {status}")
            if not success:
                all_horizons_correct = False
    
    if all_horizons_correct and horizon_results:
        print("   ✅ ALL HORIZON PREDICTIONS ARE NOW WORKING!")
    elif horizon_results:
        print("   ❌ Some horizon predictions are still broken")
    else:
        print("   ❌ Could not test horizon predictions (training failed)")
    
    # Overall result
    print(f"\n🎉 OVERALL RESULT:")
    if ensemble_metrics_fixed and all_horizons_correct and horizon_results:
        print("   ✅ ALL FIXES ARE WORKING! 🎉")
        print("   - Ensemble metrics now show proper values")
        print("   - All models return correct number of predictions for each horizon")
        print("   - 10-day forecasts now return 10 predictions (not 5)")
        print("\n   You can now use all features in the adaptive learning showcase!")
    else:
        issues = []
        if not ensemble_metrics_fixed:
            issues.append("Ensemble metrics still zero")
        if not all_horizons_correct or not horizon_results:
            issues.append("Horizon predictions still incorrect")
        
        print(f"   ❌ Some issues remain: {', '.join(issues)}")
        print("   Please check the error messages above for details.")

if __name__ == "__main__":
    main()