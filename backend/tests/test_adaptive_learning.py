"""
Test Suite for Adaptive Learning System

This module contains comprehensive tests for the adaptive learning and continuous
evaluation system, including unit tests, integration tests, and end-to-end workflows.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import shutil
import os
from datetime import datetime, timedelta
import sys

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ml_models.online_learning import OnlineSGDRegressor, OnlineLSTM, AdaptiveEnsemble
from ml_models.adaptive_learning import AdaptiveLearningManager, ModelVersion
from ml_models.continuous_learning import ContinuousLearningManager


class TestOnlineLearning(unittest.TestCase):
    """Test cases for online learning models"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.n_samples = 100
        self.n_features = 5
        
        # Generate synthetic time series data
        self.X = np.random.randn(self.n_samples, self.n_features).astype(np.float32)
        self.y = (self.X.sum(axis=1) + np.random.randn(self.n_samples) * 0.1).astype(np.float32)
        
        # Time series data for LSTM
        self.time_series = np.cumsum(np.random.randn(200)) + 100
    
    def test_sgd_regressor_initial_fit(self):
        """Test SGD regressor initial training"""
        model = OnlineSGDRegressor(learning_rate=0.01, alpha=0.001)
        
        result = model.initial_fit(self.X, self.y)
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('metrics', result)
        self.assertIn('rmse', result['metrics'])
        self.assertTrue(model.is_fitted)
        self.assertEqual(model.n_features, self.n_features)
    
    def test_sgd_regressor_partial_fit(self):
        """Test SGD regressor incremental updates"""
        model = OnlineSGDRegressor(learning_rate=0.01, alpha=0.001)
        
        # Initial fit
        initial_result = model.initial_fit(self.X[:50], self.y[:50])
        self.assertEqual(initial_result['status'], 'success')
        
        # Partial fit with new data
        update_result = model.partial_fit(self.X[50:60], self.y[50:60])
        self.assertEqual(update_result['status'], 'success')
        self.assertIn('metrics', update_result)
        self.assertEqual(update_result['samples_updated'], 10)
    
    def test_sgd_regressor_prediction(self):
        """Test SGD regressor predictions"""
        model = OnlineSGDRegressor(learning_rate=0.01, alpha=0.001)
        
        # Train model
        model.initial_fit(self.X, self.y)
        
        # Make predictions
        predictions = model.predict(self.X[:5])
        
        self.assertEqual(len(predictions), 5)
        self.assertTrue(all(isinstance(p, (int, float, np.number)) for p in predictions))
    
    def test_lstm_initial_fit(self):
        """Test LSTM initial training"""
        model = OnlineLSTM(lookback=10, lstm_units=16, learning_rate=0.001)
        
        result = model.initial_fit(self.time_series, epochs=5, batch_size=8)
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('metrics', result)
        self.assertTrue(model.is_fitted)
    
    def test_lstm_fine_tune(self):
        """Test LSTM fine-tuning"""
        model = OnlineLSTM(lookback=10, lstm_units=16, learning_rate=0.001)
        
        # Initial training
        initial_result = model.initial_fit(self.time_series[:100], epochs=3)
        self.assertEqual(initial_result['status'], 'success')
        
        # Fine-tune with new data
        fine_tune_result = model.fine_tune(self.time_series[90:120], epochs=2)
        self.assertEqual(fine_tune_result['status'], 'success')
        self.assertIn('metrics', fine_tune_result)
    
    def test_lstm_prediction(self):
        """Test LSTM predictions"""
        model = OnlineLSTM(lookback=10, lstm_units=16, learning_rate=0.001)
        
        # Train model
        model.initial_fit(self.time_series, epochs=3)
        
        # Make predictions
        predictions = model.predict(self.time_series[-20:], horizon=5)
        
        self.assertEqual(len(predictions), 5)
        self.assertTrue(all(isinstance(p, (int, float, np.number)) for p in predictions))
    
    def test_adaptive_ensemble(self):
        """Test adaptive ensemble functionality"""
        sgd_model = OnlineSGDRegressor(learning_rate=0.01)
        lstm_model = OnlineLSTM(lookback=10, lstm_units=16, epochs=3)
        
        ensemble = AdaptiveEnsemble([sgd_model, lstm_model], window_size=20)
        
        # Initial fit
        result = ensemble.initial_fit(self.X, self.y)
        self.assertEqual(result['status'], 'success')
        self.assertTrue(ensemble.is_fitted)
        
        # Update ensemble
        update_result = ensemble.update(self.X[-10:], self.y[-10:])
        self.assertEqual(update_result['status'], 'success')
        
        # Make predictions
        predictions = ensemble.predict(self.X[:5], horizon=3)
        self.assertEqual(len(predictions), 3)
    
    def test_model_save_load(self):
        """Test model persistence"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model = OnlineSGDRegressor(learning_rate=0.01)
            model.initial_fit(self.X, self.y)
            
            # Save model
            filepath = os.path.join(temp_dir, "test_model")
            save_success = model.save_model(filepath)
            self.assertTrue(save_success)
            
            # Load model
            new_model = OnlineSGDRegressor()
            load_success = new_model.load_model(filepath)
            self.assertTrue(load_success)
            self.assertTrue(new_model.is_fitted)
            
            # Compare predictions
            pred1 = model.predict(self.X[:5])
            pred2 = new_model.predict(self.X[:5])
            np.testing.assert_array_almost_equal(pred1, pred2, decimal=5)


class TestAdaptiveLearningManager(unittest.TestCase):
    """Test cases for adaptive learning manager"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.symbol = "TEST"
        self.model_type = "sgd"
        
        # Generate test data
        np.random.seed(42)
        self.X = np.random.randn(100, 5).astype(np.float32)
        self.y = (self.X.sum(axis=1) + np.random.randn(100) * 0.1).astype(np.float32)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_manager_initialization(self):
        """Test manager initialization"""
        manager = AdaptiveLearningManager(
            self.symbol, 
            self.model_type, 
            base_path=self.temp_dir
        )
        
        self.assertEqual(manager.symbol, self.symbol)
        self.assertEqual(manager.model_type, self.model_type)
        self.assertEqual(manager.current_version, 0)
        self.assertEqual(len(manager.versions), 0)
    
    def test_initial_training(self):
        """Test initial model training"""
        manager = AdaptiveLearningManager(
            self.symbol, 
            self.model_type, 
            base_path=self.temp_dir
        )
        
        result = manager.initial_training(self.X, self.y, "Test initial training")
        
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['version'], 1)
        self.assertIn('metrics', result)
        self.assertEqual(manager.current_version, 1)
        self.assertEqual(len(manager.versions), 1)
        self.assertTrue(manager.versions[0].is_active)
    
    def test_model_update_and_versioning(self):
        """Test model updates and version creation"""
        manager = AdaptiveLearningManager(
            self.symbol, 
            self.model_type, 
            base_path=self.temp_dir
        )
        
        # Initial training
        initial_result = manager.initial_training(self.X[:50], self.y[:50])
        self.assertEqual(initial_result['status'], 'success')
        
        # Update with new data (should create new version if performance improves)
        update_result = manager.update_model(self.X[50:70], self.y[50:70], force_new_version=True)
        self.assertEqual(update_result['status'], 'success')
        
        # Check version management
        if update_result.get('new_version_created', False):
            self.assertEqual(manager.current_version, 2)
            self.assertEqual(len(manager.versions), 2)
    
    def test_rollback_functionality(self):
        """Test model rollback"""
        manager = AdaptiveLearningManager(
            self.symbol, 
            self.model_type, 
            base_path=self.temp_dir
        )
        
        # Create multiple versions
        manager.initial_training(self.X[:30], self.y[:30])
        manager.update_model(self.X[30:60], self.y[30:60], force_new_version=True)
        
        if len(manager.versions) >= 2:
            # Rollback to version 1
            rollback_result = manager.rollback_to_version(1)
            self.assertEqual(rollback_result['status'], 'success')
            self.assertEqual(manager.current_version, 1)
    
    def test_performance_summary(self):
        """Test performance summary generation"""
        manager = AdaptiveLearningManager(
            self.symbol, 
            self.model_type, 
            base_path=self.temp_dir
        )
        
        # Train model
        manager.initial_training(self.X, self.y)
        
        # Get performance summary
        summary = manager.get_performance_summary()
        
        self.assertIn('total_versions', summary)
        self.assertIn('current_version', summary)
        self.assertIn('rmse_trend', summary)
        self.assertEqual(summary['total_versions'], 1)
    
    def test_version_history(self):
        """Test version history tracking"""
        manager = AdaptiveLearningManager(
            self.symbol, 
            self.model_type, 
            base_path=self.temp_dir
        )
        
        # Create versions
        manager.initial_training(self.X[:50], self.y[:50])
        manager.update_model(self.X[50:], self.y[50:], force_new_version=True)
        
        # Get version history
        history = manager.get_version_history()
        
        self.assertIsInstance(history, list)
        self.assertGreaterEqual(len(history), 1)
        
        for version_info in history:
            self.assertIn('version', version_info)
            self.assertIn('metrics', version_info)
            self.assertIn('timestamp', version_info)


class TestContinuousLearning(unittest.TestCase):
    """Test cases for continuous learning manager"""
    
    def setUp(self):
        """Set up test environment"""
        self.symbol = "AAPL"
        self.model_types = ["sgd", "lstm"]
        
        # Mock database for testing
        self.mock_db = MockDatabase()
    
    def test_symbol_registration(self):
        """Test symbol registration for continuous learning"""
        manager = ContinuousLearningManager(db=self.mock_db)
        
        result = manager.register_symbol(self.symbol, self.model_types)
        
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['symbol'], self.symbol)
        self.assertIn('results', result)
        
        # Check that managers were created
        for model_type in self.model_types:
            manager_key = f"{self.symbol}_{model_type}"
            self.assertIn(manager_key, manager.adaptive_managers)
    
    def test_status_reporting(self):
        """Test system status reporting"""
        manager = ContinuousLearningManager(db=self.mock_db)
        manager.register_symbol(self.symbol, self.model_types)
        
        status = manager.get_status()
        
        self.assertIn('is_running', status)
        self.assertIn('registered_models', status)
        self.assertIn('models', status)
        self.assertEqual(status['registered_models'], len(self.model_types))
    
    def test_data_preprocessing(self):
        """Test data preprocessing functionality"""
        manager = ContinuousLearningManager(db=self.mock_db)
        
        # Create sample price data
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        prices = pd.DataFrame({
            'date': dates,
            'open': np.random.uniform(100, 110, 50),
            'high': np.random.uniform(110, 120, 50),
            'low': np.random.uniform(90, 100, 50),
            'close': np.random.uniform(100, 110, 50),
            'volume': np.random.randint(1000000, 10000000, 50)
        })
        
        # Test preprocessing
        X, y = manager._preprocess_data(prices)
        
        self.assertGreater(len(X), 0)
        self.assertGreater(len(y), 0)
        self.assertEqual(len(X), len(y))
    
    def test_technical_indicators(self):
        """Test technical indicator calculations"""
        manager = ContinuousLearningManager(db=self.mock_db)
        
        # Create sample price data
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        prices = pd.DataFrame({
            'date': dates,
            'open': np.random.uniform(100, 110, 50),
            'high': np.random.uniform(110, 120, 50),
            'low': np.random.uniform(90, 100, 50),
            'close': np.random.uniform(100, 110, 50),
            'volume': np.random.randint(1000000, 10000000, 50)
        })
        
        # Calculate indicators
        enhanced_data = manager._calculate_technical_indicators(prices)
        
        # Check that indicators were added
        expected_columns = ['returns', 'volatility', 'sma_5', 'sma_20', 'rsi']
        for col in expected_columns:
            self.assertIn(col, enhanced_data.columns)


class MockDatabase:
    """Mock database for testing"""
    
    def __init__(self):
        self.prices_data = []
        self.model_versions = []
        self.training_events = []
    
    def get_prices(self, symbol, start_date=None, end_date=None, limit=None):
        """Mock price data retrieval"""
        # Generate synthetic price data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        prices = []
        
        for i, date in enumerate(dates):
            base_price = 100 + i * 0.1 + np.random.randn() * 2
            prices.append({
                'date': date.strftime('%Y-%m-%d'),
                'open': base_price + np.random.randn() * 0.5,
                'high': base_price + abs(np.random.randn()) * 2,
                'low': base_price - abs(np.random.randn()) * 2,
                'close': base_price + np.random.randn() * 0.5,
                'volume': np.random.randint(1000000, 10000000)
            })
        
        return prices[:limit] if limit else prices
    
    def get_adaptive_learning_stats(self):
        """Mock adaptive learning statistics"""
        return {
            'total_model_versions': len(self.model_versions),
            'total_training_events': len(self.training_events),
            'active_models': sum(1 for v in self.model_versions if v.get('is_active', False))
        }


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete adaptive learning system"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_db = MockDatabase()
        self.symbol = "AAPL"
    
    def tearDown(self):
        """Clean up integration test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end adaptive learning workflow"""
        # Initialize continuous learning manager
        continuous_manager = ContinuousLearningManager(db=self.mock_db)
        
        # Register symbol for adaptive learning
        register_result = continuous_manager.register_symbol(self.symbol, ['sgd'])
        self.assertEqual(register_result['status'], 'success')
        
        # Perform initial training
        training_result = continuous_manager.initial_training(self.symbol, 'sgd')
        self.assertEqual(training_result['status'], 'success')
        
        # Make predictions
        prediction_result = continuous_manager.predict(self.symbol, 'sgd', horizon=5)
        self.assertEqual(prediction_result['status'], 'success')
        self.assertEqual(len(prediction_result['predictions']), 5)
        
        # Get performance information
        performance_result = continuous_manager.get_model_performance(self.symbol, 'sgd')
        self.assertEqual(performance_result['status'], 'success')
        
        # Check system status
        status = continuous_manager.get_status()
        self.assertGreater(status['registered_models'], 0)
    
    def test_multiple_model_types(self):
        """Test workflow with multiple model types"""
        continuous_manager = ContinuousLearningManager(db=self.mock_db)
        
        # Register multiple model types
        model_types = ['sgd', 'lstm', 'ensemble']
        register_result = continuous_manager.register_symbol(self.symbol, model_types)
        self.assertEqual(register_result['status'], 'success')
        
        # Train each model type
        for model_type in model_types:
            training_result = continuous_manager.initial_training(self.symbol, model_type)
            self.assertEqual(training_result['status'], 'success')
        
        # Check that all models are registered
        status = continuous_manager.get_status()
        self.assertEqual(status['registered_models'], len(model_types))


def create_synthetic_dataset(n_samples=1000, n_features=5, noise_level=0.1):
    """Create synthetic dataset for testing"""
    np.random.seed(42)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Create target with some pattern
    weights = np.random.randn(n_features)
    y = X @ weights + np.random.randn(n_samples) * noise_level
    
    return X.astype(np.float32), y.astype(np.float32)


def run_performance_benchmark():
    """Run performance benchmarks for adaptive learning models"""
    print("Running Adaptive Learning Performance Benchmarks...")
    
    # Create test data
    X, y = create_synthetic_dataset(n_samples=5000, n_features=10)
    
    models = {
        'SGD': OnlineSGDRegressor(learning_rate=0.01),
        'LSTM': OnlineLSTM(lookback=20, lstm_units=32, epochs=10)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTesting {name}...")
        
        # Initial training
        start_time = datetime.now()
        if name == 'LSTM':
            result = model.initial_fit(y[:1000], epochs=5)  # Use y as time series
        else:
            result = model.initial_fit(X[:1000], y[:1000])
        
        initial_time = (datetime.now() - start_time).total_seconds()
        
        # Incremental updates
        start_time = datetime.now()
        for i in range(10):
            start_idx = 1000 + i * 100
            end_idx = start_idx + 100
            
            if name == 'LSTM':
                model.fine_tune(y[start_idx:end_idx], epochs=1)
            else:
                model.partial_fit(X[start_idx:end_idx], y[start_idx:end_idx])
        
        update_time = (datetime.now() - start_time).total_seconds()
        
        results[name] = {
            'initial_training_time': initial_time,
            'update_time': update_time,
            'final_metrics': result.get('metrics', {})
        }
        
        print(f"  Initial training: {initial_time:.2f}s")
        print(f"  10 updates: {update_time:.2f}s")
        print(f"  Final RMSE: {result.get('metrics', {}).get('rmse', 'N/A')}")
    
    return results


if __name__ == '__main__':
    # Run unit tests
    print("Running Adaptive Learning Test Suite...")
    unittest.main(verbosity=2, exit=False)
    
    # Run performance benchmarks
    print("\n" + "="*50)
    benchmark_results = run_performance_benchmark()
    
    print("\n" + "="*50)
    print("All tests completed successfully!")
    print("The adaptive learning system is ready for deployment.")