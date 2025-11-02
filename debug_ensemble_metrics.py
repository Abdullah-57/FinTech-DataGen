#!/usr/bin/env python3
"""
Debug script to test ensemble metrics calculation
"""

import numpy as np
import sys
import os

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from ml_models.online_learning import OnlineSGDRegressor, OnlineLSTM, AdaptiveEnsemble

def test_ensemble_metrics():
    """Test ensemble metrics calculation"""
    print("🧪 Testing Ensemble Metrics Calculation")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(50, 10).astype(np.float32)
    y = np.random.randn(50).astype(np.float32) * 10 + 100  # Stock-like prices
    
    print(f"Sample data shape: X={X.shape}, y={y.shape}")
    print(f"Sample y values: {y[:5]}")
    
    # Create individual models
    sgd_model = OnlineSGDRegressor(learning_rate=0.01, alpha=0.0001)
    lstm_model = OnlineLSTM(lookback=10, lstm_units=16, learning_rate=0.001)
    
    # Create ensemble
    ensemble = AdaptiveEnsemble([sgd_model, lstm_model], window_size=20)
    
    print("\n🔄 Testing initial_fit...")
    result = ensemble.initial_fit(X, y)
    
    print(f"Initial fit result: {result}")
    
    if result['status'] == 'success':
        metrics = result.get('metrics', {})
        print(f"\n📊 Ensemble Metrics:")
        print(f"   RMSE: {metrics.get('rmse', 'N/A')}")
        print(f"   MAE: {metrics.get('mae', 'N/A')}")
        print(f"   MAPE: {metrics.get('mape', 'N/A')}")
        
        # Check if metrics are non-zero
        has_real_metrics = any(metrics.get(m, 0) != 0.0 for m in ['rmse', 'mae', 'mape'])
        if has_real_metrics:
            print("   ✅ Metrics are properly calculated!")
        else:
            print("   ❌ Metrics are all zero - investigating...")
            
            # Test the _calculate_ensemble_metrics method directly
            print("\n🔍 Testing _calculate_ensemble_metrics directly...")
            try:
                # Make a prediction to test
                test_X = X[:5]
                predictions = ensemble.predict(test_X, horizon=1)
                actual_values = y[:5]
                
                print(f"Test predictions: {predictions}")
                print(f"Test actual values: {actual_values}")
                
                if len(predictions) == len(actual_values):
                    direct_metrics = ensemble._calculate_ensemble_metrics(actual_values, predictions)
                    print(f"Direct metrics calculation: {direct_metrics}")
                else:
                    print(f"Length mismatch: predictions={len(predictions)}, actual={len(actual_values)}")
                    
            except Exception as e:
                print(f"Error in direct metrics test: {e}")
                import traceback
                traceback.print_exc()
    else:
        print(f"❌ Initial fit failed: {result.get('message', 'Unknown error')}")
    
    # Test update method
    print("\n🔄 Testing update...")
    update_result = ensemble.update(X[-10:], y[-10:])
    
    print(f"Update result: {update_result}")
    
    if update_result['status'] == 'success':
        update_metrics = update_result.get('metrics', {})
        print(f"\n📊 Update Metrics:")
        print(f"   RMSE: {update_metrics.get('rmse', 'N/A')}")
        print(f"   MAE: {update_metrics.get('mae', 'N/A')}")
        print(f"   MAPE: {update_metrics.get('mape', 'N/A')}")

def test_horizon_predictions():
    """Test horizon predictions"""
    print("\n\n🎯 Testing Horizon Predictions")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(50, 10).astype(np.float32)
    y = np.random.randn(50).astype(np.float32) * 10 + 100
    
    # Test SGD
    print("\n🔄 Testing SGD horizon predictions...")
    sgd_model = OnlineSGDRegressor(learning_rate=0.01)
    sgd_result = sgd_model.initial_fit(X, y)
    
    if sgd_result['status'] == 'success':
        for horizon in [1, 5, 10]:
            predictions = sgd_model.predict(X[-1:], horizon=horizon)
            print(f"   SGD {horizon}-day: {len(predictions)} predictions")
            if len(predictions) != horizon:
                print(f"   ❌ Expected {horizon}, got {len(predictions)}")
            else:
                print(f"   ✅ Correct horizon length")
    
    # Test LSTM
    print("\n🔄 Testing LSTM horizon predictions...")
    lstm_model = OnlineLSTM(lookback=10, lstm_units=16, epochs=3)
    lstm_result = lstm_model.initial_fit(y, epochs=3)  # LSTM uses y only
    
    if lstm_result['status'] == 'success':
        for horizon in [1, 5, 10]:
            predictions = lstm_model.predict(y[-20:], horizon=horizon)
            print(f"   LSTM {horizon}-day: {len(predictions)} predictions")
            if len(predictions) != horizon:
                print(f"   ❌ Expected {horizon}, got {len(predictions)}")
            else:
                print(f"   ✅ Correct horizon length")
    
    # Test Ensemble
    print("\n🔄 Testing Ensemble horizon predictions...")
    sgd_model2 = OnlineSGDRegressor(learning_rate=0.01)
    lstm_model2 = OnlineLSTM(lookback=10, lstm_units=16, epochs=3)
    ensemble = AdaptiveEnsemble([sgd_model2, lstm_model2], window_size=20)
    
    ensemble_result = ensemble.initial_fit(X, y)
    
    if ensemble_result['status'] == 'success':
        for horizon in [1, 5, 10]:
            predictions = ensemble.predict(X[-1:], horizon=horizon)
            print(f"   Ensemble {horizon}-day: {len(predictions)} predictions")
            if len(predictions) != horizon:
                print(f"   ❌ Expected {horizon}, got {len(predictions)}")
            else:
                print(f"   ✅ Correct horizon length")

if __name__ == "__main__":
    test_ensemble_metrics()
    test_horizon_predictions()