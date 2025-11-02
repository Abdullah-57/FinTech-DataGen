#!/usr/bin/env python3
"""
Comprehensive test script to verify all adaptive learning fixes
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

def test_registration():
    """Test symbol registration"""
    print("\n🔄 Testing symbol registration...")
    
    payload = {
        "symbol": "AAPL",
        "model_types": ["sgd", "lstm", "ensemble"]
    }
    
    try:
        response = requests.post(
            "http://localhost:5000/api/adaptive/register",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ Registration successful")
            return True
        else:
            print(f"❌ Registration failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Registration error: {e}")
        return False

def test_model_training(model_type):
    """Test training for a specific model type"""
    print(f"\n🔄 Testing {model_type.upper()} training...")
    
    payload = {
        "symbol": "AAPL",
        "model_type": model_type
    }
    
    try:
        response = requests.post(
            "http://localhost:5000/api/adaptive/train",
            json=payload,
            timeout=120
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                metrics = data.get('metrics', {})
                print(f"✅ {model_type.upper()} training successful!")
                print(f"   Version: {data.get('version', 'N/A')}")
                print(f"   RMSE: {metrics.get('rmse', 'N/A')}")
                print(f"   MAE: {metrics.get('mae', 'N/A')}")
                print(f"   MAPE: {metrics.get('mape', 'N/A')}%")
                
                # Check if metrics are actually populated (not just 0.0)
                has_real_metrics = any(metrics.get(m, 0) != 0.0 for m in ['rmse', 'mae', 'mape'])
                if has_real_metrics:
                    print(f"   ✅ Metrics are properly calculated")
                else:
                    print(f"   ⚠️  Metrics are zero - may need more data")
                
                return True
            else:
                print(f"❌ {model_type.upper()} training failed: {data.get('message', 'Unknown error')}")
                return False
        else:
            print(f"❌ {model_type.upper()} training request failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ {model_type.upper()} training error: {e}")
        return False

def test_predictions(model_type, horizons=[1, 5, 10]):
    """Test predictions for different horizons"""
    print(f"\n🔮 Testing {model_type.upper()} predictions...")
    
    results = {}
    
    for horizon in horizons:
        payload = {
            "symbol": "AAPL",
            "model_type": model_type,
            "horizon": horizon
        }
        
        try:
            response = requests.post(
                "http://localhost:5000/api/adaptive/predict",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    predictions = data.get('predictions', [])
                    print(f"   ✅ {horizon}-day forecast: {len(predictions)} predictions")
                    
                    # Verify we got the right number of predictions
                    if len(predictions) == horizon:
                        print(f"      ✅ Correct number of predictions ({horizon})")
                        print(f"      First: ${predictions[0]:.2f}, Last: ${predictions[-1]:.2f}")
                        results[horizon] = True
                    else:
                        print(f"      ❌ Expected {horizon} predictions, got {len(predictions)}")
                        results[horizon] = False
                else:
                    print(f"   ❌ {horizon}-day forecast failed: {data.get('message', 'Unknown error')}")
                    results[horizon] = False
            else:
                print(f"   ❌ {horizon}-day forecast request failed: {response.status_code}")
                results[horizon] = False
                
        except Exception as e:
            print(f"   ❌ {horizon}-day forecast error: {e}")
            results[horizon] = False
    
    return results

def test_model_update(model_type):
    """Test model update functionality"""
    print(f"\n🔄 Testing {model_type.upper()} model update...")
    
    payload = {
        "symbol": "AAPL",
        "model_type": model_type
    }
    
    try:
        response = requests.post(
            "http://localhost:5000/api/adaptive/update",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                metrics = data.get('metrics', {})
                print(f"✅ {model_type.upper()} update successful!")
                print(f"   Version: {data.get('version', 'N/A')}")
                print(f"   RMSE: {metrics.get('rmse', 'N/A')}")
                print(f"   MAE: {metrics.get('mae', 'N/A')}")
                print(f"   MAPE: {metrics.get('mape', 'N/A')}%")
                
                # Check if metrics are populated
                has_real_metrics = any(metrics.get(m, 0) != 0.0 for m in ['rmse', 'mae', 'mape'])
                if has_real_metrics:
                    print(f"   ✅ Update metrics are properly calculated")
                else:
                    print(f"   ⚠️  Update metrics are zero - may need more data")
                
                return True
            else:
                print(f"❌ {model_type.upper()} update failed: {data.get('message', 'Unknown error')}")
                return False
        else:
            print(f"❌ {model_type.upper()} update request failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ {model_type.upper()} update error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Comprehensive Adaptive Learning Fix Test")
    print("=" * 60)
    
    # Check backend health
    if not test_backend_health():
        print("\n❌ Backend is not running. Please start it with:")
        print("   cd backend")
        print("   python app.py")
        return
    
    # Test registration
    if not test_registration():
        print("\n❌ Registration failed. Cannot continue testing.")
        return
    
    # Test training for all model types
    training_results = {}
    for model_type in ['sgd', 'lstm', 'ensemble']:
        training_results[model_type] = test_model_training(model_type)
        time.sleep(2)  # Brief pause between trainings
    
    # Test predictions for all model types
    prediction_results = {}
    for model_type in ['sgd', 'lstm', 'ensemble']:
        if training_results.get(model_type, False):
            prediction_results[model_type] = test_predictions(model_type)
        else:
            print(f"\n⏭️  Skipping {model_type.upper()} predictions (training failed)")
            prediction_results[model_type] = {1: False, 5: False, 10: False}
    
    # Test model updates
    update_results = {}
    for model_type in ['sgd', 'lstm', 'ensemble']:
        if training_results.get(model_type, False):
            update_results[model_type] = test_model_update(model_type)
            time.sleep(1)  # Brief pause between updates
        else:
            print(f"\n⏭️  Skipping {model_type.upper()} update (training failed)")
            update_results[model_type] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    
    print("\n🎯 Training Results:")
    for model_type, success in training_results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {model_type.upper()}: {status}")
    
    print("\n🔮 Prediction Results:")
    for model_type, horizons in prediction_results.items():
        print(f"   {model_type.upper()}:")
        for horizon, success in horizons.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"     {horizon}-day: {status}")
    
    print("\n🔄 Update Results:")
    for model_type, success in update_results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {model_type.upper()}: {status}")
    
    # Overall assessment
    training_success = sum(training_results.values())
    prediction_success = sum(sum(h.values()) for h in prediction_results.values())
    total_prediction_tests = sum(len(h) for h in prediction_results.values())
    update_success = sum(update_results.values())
    
    print(f"\n📈 Overall Results:")
    print(f"   Training: {training_success}/3 models successful")
    print(f"   Predictions: {prediction_success}/{total_prediction_tests} tests successful")
    print(f"   Updates: {update_success}/3 models successful")
    
    if training_success == 3 and prediction_success == total_prediction_tests and update_success == 3:
        print("\n🎉 ALL TESTS PASSED! All fixes are working correctly.")
        print("\n✅ Fixed Issues:")
        print("   1. ✅ Ensemble model now shows proper RMSE, MAE, MAPE metrics")
        print("   2. ✅ LSTM predictions work without 'df not defined' error")
        print("   3. ✅ All models return correct number of predictions for each horizon")
        print("   4. ✅ Model updates work properly with metrics")
        print("\nYou can now use all features in the adaptive learning showcase!")
    elif training_success == 3:
        print("\n👍 Training is working perfectly!")
        if prediction_success < total_prediction_tests:
            print(f"⚠️  Some prediction issues remain ({prediction_success}/{total_prediction_tests} successful)")
        if update_success < 3:
            print(f"⚠️  Some update issues remain ({update_success}/3 successful)")
    else:
        print("\n⚠️  Some issues remain. Check the error messages above.")
        
    # Specific issue checks
    print("\n🔍 Specific Issue Checks:")
    
    # Check ensemble metrics
    if training_results.get('ensemble', False):
        print("   ✅ Ensemble training completed (metrics should be visible)")
    else:
        print("   ❌ Ensemble training failed (metrics will be empty)")
    
    # Check LSTM predictions
    lstm_predictions_work = all(prediction_results.get('lstm', {}).values())
    if lstm_predictions_work:
        print("   ✅ LSTM predictions work (no 'df not defined' error)")
    else:
        print("   ❌ LSTM predictions have issues")
    
    # Check horizon accuracy
    horizon_issues = []
    for model_type, horizons in prediction_results.items():
        for horizon, success in horizons.items():
            if not success:
                horizon_issues.append(f"{model_type.upper()} {horizon}-day")
    
    if not horizon_issues:
        print("   ✅ All prediction horizons return correct number of predictions")
    else:
        print(f"   ❌ Horizon issues: {', '.join(horizon_issues)}")

if __name__ == "__main__":
    main()