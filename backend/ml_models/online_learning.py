"""
Online Learning Models for Adaptive Financial Forecasting

This module implements online learning algorithms that can update incrementally
as new data arrives, supporting continuous adaptation to market changes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class OnlineSGDRegressor:
    """
    Online SGD Regressor with incremental learning capabilities.
    Fast updates suitable for high-frequency data streams.
    """
    
    def __init__(self, learning_rate: float = 0.01, alpha: float = 0.0001):
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.model = SGDRegressor(
            learning_rate='constant',
            eta0=learning_rate,
            alpha=alpha,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.n_features = None
        self.update_count = 0
        
    def initial_fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Initial training on historical data"""
        try:
            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.float32)
            
            if len(X) == 0:
                raise ValueError("Empty training data")
                
            # Fit scaler and transform data
            X_scaled = self.scaler.fit_transform(X)
            
            # Initial fit
            self.model.fit(X_scaled, y)
            self.is_fitted = True
            self.n_features = X.shape[1]
            
            # Calculate initial metrics
            y_pred = self.model.predict(X_scaled)
            metrics = self._calculate_metrics(y, y_pred)
            
            return {
                'status': 'success',
                'metrics': metrics,
                'samples_trained': len(X),
                'features': self.n_features
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Incremental update with new data"""
        try:
            if not self.is_fitted:
                return self.initial_fit(X, y)
                
            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.float32)
            
            if X.shape[1] != self.n_features:
                raise ValueError(f"Feature mismatch: expected {self.n_features}, got {X.shape[1]}")
            
            # Transform new data using existing scaler
            X_scaled = self.scaler.transform(X)
            
            # Partial fit (incremental update)
            self.model.partial_fit(X_scaled, y)
            self.update_count += len(X)
            
            # Calculate metrics on new data
            y_pred = self.model.predict(X_scaled)
            metrics = self._calculate_metrics(y, y_pred)
            
            return {
                'status': 'success',
                'metrics': metrics,
                'samples_updated': len(X),
                'total_updates': self.update_count
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def predict(self, X: np.ndarray, horizon: int = 1) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
            
        X = np.array(X, dtype=np.float32)
        X_scaled = self.scaler.transform(X)
        
        # For SGD, we can only predict one step ahead, so repeat the prediction
        base_prediction = self.model.predict(X_scaled)
        
        if horizon == 1:
            return base_prediction
        else:
            # For multi-step prediction, repeat the last prediction with small variations
            predictions = []
            last_pred = base_prediction[-1] if len(base_prediction) > 0 else 0.0
            
            for i in range(horizon):
                # Add small random variation to simulate future uncertainty
                variation = np.random.normal(0, abs(last_pred) * 0.01)  # 1% variation
                pred = last_pred + variation
                predictions.append(pred)
                last_pred = pred
            
            return np.array(predictions)
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics"""
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        
        # Avoid division by zero in MAPE
        nonzero_mask = y_true != 0
        if np.any(nonzero_mask):
            mape = float(np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100)
        else:
            mape = 0.0
            
        return {'rmse': rmse, 'mae': mae, 'mape': mape}
    
    def save_model(self, filepath: str) -> bool:
        """Save model to disk"""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'is_fitted': self.is_fitted,
                'n_features': self.n_features,
                'update_count': self.update_count,
                'learning_rate': self.learning_rate,
                'alpha': self.alpha
            }
            joblib.dump(model_data, filepath)
            return True
        except Exception as e:
            print(f"Error saving SGD model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load model from disk"""
        try:
            if not os.path.exists(filepath):
                return False
                
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.is_fitted = model_data['is_fitted']
            self.n_features = model_data['n_features']
            self.update_count = model_data['update_count']
            self.learning_rate = model_data['learning_rate']
            self.alpha = model_data['alpha']
            return True
        except Exception as e:
            print(f"Error loading SGD model: {e}")
            return False


class OnlineLSTM:
    """
    Online LSTM with incremental fine-tuning capabilities.
    Suitable for complex temporal pattern learning with periodic updates.
    """
    
    def __init__(self, lookback: int = 10, lstm_units: int = 32, 
                 learning_rate: float = 0.001, fine_tune_lr: float = 0.0001):
        self.lookback = lookback
        self.lstm_units = lstm_units
        self.learning_rate = learning_rate
        self.fine_tune_lr = fine_tune_lr
        
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.n_features = None
        self.update_count = 0
        self.min_val = 0.0
        self.max_val = 1.0
        
    def _build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(self.lstm_units, input_shape=input_shape, return_sequences=False),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model
    
    def _prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training"""
        X, y = [], []
        for i in range(len(data) - self.lookback):
            X.append(data[i:(i + self.lookback)])
            y.append(data[i + self.lookback])
        return np.array(X), np.array(y)
    
    def _scale_data(self, data: np.ndarray, fit_scaler: bool = False) -> np.ndarray:
        """Scale data for neural network"""
        if fit_scaler:
            self.min_val = float(np.min(data))
            self.max_val = float(np.max(data))
        
        range_val = self.max_val - self.min_val
        if range_val == 0:
            range_val = 1.0
            
        return (data - self.min_val) / range_val
    
    def _inverse_scale(self, data: np.ndarray) -> np.ndarray:
        """Inverse scale predictions"""
        range_val = self.max_val - self.min_val
        if range_val == 0:
            range_val = 1.0
        return data * range_val + self.min_val
    
    def initial_fit(self, data: np.ndarray, epochs: int = 50, batch_size: int = 16) -> Dict[str, Any]:
        """Initial training on historical data"""
        try:
            data = np.array(data, dtype=np.float32).flatten()
            
            if len(data) < self.lookback + 10:
                raise ValueError(f"Insufficient data: need at least {self.lookback + 10} samples")
            
            # Scale data
            scaled_data = self._scale_data(data, fit_scaler=True)
            
            # Prepare sequences
            X, y = self._prepare_sequences(scaled_data)
            
            if len(X) == 0:
                raise ValueError("No sequences could be created")
            
            # Reshape for LSTM
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            # Build and train model
            self.model = self._build_model((self.lookback, 1))
            
            # Ensure epochs is an integer
            epochs = int(epochs) if epochs is not None else 50
            batch_size = int(batch_size) if batch_size is not None else 16
            
            # Train with early stopping
            history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                verbose=0,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True
                    )
                ]
            )
            
            self.is_fitted = True
            self.n_features = 1
            
            # Calculate metrics
            y_pred = self.model.predict(X, verbose=0).flatten()
            y_pred_scaled = self._inverse_scale(y_pred)
            y_true_scaled = self._inverse_scale(y)
            metrics = self._calculate_metrics(y_true_scaled, y_pred_scaled)
            
            return {
                'status': 'success',
                'metrics': metrics,
                'samples_trained': len(X),
                'epochs_trained': len(history.history['loss']),
                'final_loss': float(history.history['loss'][-1])
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def fine_tune(self, new_data: np.ndarray, epochs: int = 5) -> Dict[str, Any]:
        """Fine-tune model with new data"""
        try:
            if not self.is_fitted:
                return self.initial_fit(new_data, epochs=epochs*2)
            
            new_data = np.array(new_data, dtype=np.float32).flatten()
            
            if len(new_data) < self.lookback + 1:
                raise ValueError(f"Insufficient new data: need at least {self.lookback + 1} samples")
            
            # Scale new data using existing parameters
            scaled_data = self._scale_data(new_data, fit_scaler=False)
            
            # Prepare sequences
            X, y = self._prepare_sequences(scaled_data)
            
            if len(X) == 0:
                raise ValueError("No sequences could be created from new data")
            
            # Reshape for LSTM
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            # Ensure epochs is an integer
            epochs = int(epochs) if epochs is not None else 5
            
            # Reduce learning rate for fine-tuning
            self.model.optimizer.learning_rate.assign(self.fine_tune_lr)
            
            # Fine-tune with new data
            history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=min(16, len(X)),
                verbose=0
            )
            
            self.update_count += len(X)
            
            # Calculate metrics on new data
            y_pred = self.model.predict(X, verbose=0).flatten()
            y_pred_scaled = self._inverse_scale(y_pred)
            y_true_scaled = self._inverse_scale(y)
            metrics = self._calculate_metrics(y_true_scaled, y_pred_scaled)
            
            return {
                'status': 'success',
                'metrics': metrics,
                'samples_updated': len(X),
                'total_updates': self.update_count,
                'epochs_fine_tuned': epochs,
                'final_loss': float(history.history['loss'][-1])
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def predict(self, data: np.ndarray, horizon: int = 1) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        data = np.array(data, dtype=np.float32).flatten()
        
        if len(data) < self.lookback:
            # Pad with last value if insufficient data
            padding = np.full(self.lookback - len(data), data[-1] if len(data) > 0 else 0.0)
            data = np.concatenate([padding, data])
        
        # Use last lookback values
        sequence = data[-self.lookback:]
        scaled_sequence = self._scale_data(sequence, fit_scaler=False)
        
        predictions = []
        current_sequence = scaled_sequence.copy()
        
        for _ in range(horizon):
            # Reshape for prediction
            X = current_sequence.reshape((1, self.lookback, 1))
            
            # Predict next value
            pred_scaled = self.model.predict(X, verbose=0)[0, 0]
            pred = self._inverse_scale(np.array([pred_scaled]))[0]
            predictions.append(pred)
            
            # Update sequence for next prediction
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = pred_scaled
        
        return np.array(predictions)
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics"""
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        
        # Avoid division by zero in MAPE
        nonzero_mask = y_true != 0
        if np.any(nonzero_mask):
            mape = float(np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100)
        else:
            mape = 0.0
            
        return {'rmse': rmse, 'mae': mae, 'mape': mape}
    
    def save_model(self, filepath: str) -> bool:
        """Save model to disk"""
        try:
            if self.model is None:
                return False
                
            # Save model architecture and weights
            model_dir = os.path.dirname(filepath)
            os.makedirs(model_dir, exist_ok=True)
            
            self.model.save(f"{filepath}_model.h5")
            
            # Save additional parameters
            params = {
                'lookback': self.lookback,
                'lstm_units': self.lstm_units,
                'learning_rate': self.learning_rate,
                'fine_tune_lr': self.fine_tune_lr,
                'is_fitted': self.is_fitted,
                'n_features': self.n_features,
                'update_count': self.update_count,
                'min_val': self.min_val,
                'max_val': self.max_val
            }
            joblib.dump(params, f"{filepath}_params.pkl")
            return True
            
        except Exception as e:
            print(f"Error saving LSTM model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load model from disk"""
        try:
            model_path = f"{filepath}_model.h5"
            params_path = f"{filepath}_params.pkl"
            
            if not (os.path.exists(model_path) and os.path.exists(params_path)):
                return False
            
            # Load model
            self.model = tf.keras.models.load_model(model_path)
            
            # Load parameters
            params = joblib.load(params_path)
            self.lookback = params['lookback']
            self.lstm_units = params['lstm_units']
            self.learning_rate = params['learning_rate']
            self.fine_tune_lr = params['fine_tune_lr']
            self.is_fitted = params['is_fitted']
            self.n_features = params['n_features']
            self.update_count = params['update_count']
            self.min_val = params['min_val']
            self.max_val = params['max_val']
            
            return True
            
        except Exception as e:
            print(f"Error loading LSTM model: {e}")
            return False


class AdaptiveEnsemble:
    """
    Adaptive ensemble that dynamically weights models based on recent performance.
    Combines predictions from multiple online learning models.
    """
    
    def __init__(self, models: List[Any], window_size: int = 50):
        self.models = models
        self.window_size = window_size
        self.model_names = [f"model_{i}" for i in range(len(models))]
        self.performance_history = {name: [] for name in self.model_names}
        self.weights = np.ones(len(models)) / len(models)  # Equal weights initially
        self.is_fitted = False
        
    def initial_fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Initial training of all models"""
        results = {}
        
        for i, model in enumerate(self.models):
            model_name = self.model_names[i]
            
            if hasattr(model, 'initial_fit'):
                result = model.initial_fit(X, y)
            else:
                # Fallback for models without initial_fit method
                try:
                    model.fit(X, y)
                    result = {'status': 'success', 'metrics': {'rmse': 0.0, 'mae': 0.0, 'mape': 0.0}}
                except Exception as e:
                    result = {'status': 'error', 'message': str(e)}
            
            results[model_name] = result
            
            # Initialize performance history
            if result['status'] == 'success' and 'metrics' in result:
                self.performance_history[model_name].append(result['metrics']['rmse'])
        
        self.is_fitted = True
        self._update_weights()
        
        # Calculate ensemble metrics by making predictions on training data
        ensemble_metrics = {'rmse': 0.0, 'mae': 0.0, 'mape': 0.0}
        try:
            if len(X) > 0 and self.is_fitted:
                # Use individual samples for metrics calculation
                test_size = min(5, len(X))
                predictions_list = []
                actual_list = []
                
                for i in range(test_size):
                    # Make prediction for single sample
                    single_X = X[i:i+1]  # Keep as 2D array
                    single_y = y[i]
                    
                    pred = self.predict(single_X, horizon=1)
                    if len(pred) > 0:
                        predictions_list.append(pred[0])  # Take first prediction
                        actual_list.append(single_y)
                
                if len(predictions_list) > 0 and len(predictions_list) == len(actual_list):
                    ensemble_metrics = self._calculate_ensemble_metrics(
                        np.array(actual_list), 
                        np.array(predictions_list)
                    )
                    print(f"Ensemble metrics calculated: RMSE={ensemble_metrics['rmse']:.4f}, MAE={ensemble_metrics['mae']:.4f}, MAPE={ensemble_metrics['mape']:.2f}%")
                else:
                    print(f"Could not calculate metrics: predictions={len(predictions_list)}, actual={len(actual_list)}")
        except Exception as e:
            print(f"Warning: Could not calculate ensemble metrics: {e}")
            import traceback
            traceback.print_exc()
        
        return {
            'status': 'success',
            'model_results': results,
            'initial_weights': self.weights.tolist(),
            'metrics': ensemble_metrics
        }
    
    def update(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Update all models and recompute weights"""
        if not self.is_fitted:
            return self.initial_fit(X, y)
        
        results = {}
        
        for i, model in enumerate(self.models):
            model_name = self.model_names[i]
            
            try:
                # Update model
                if hasattr(model, 'partial_fit'):
                    result = model.partial_fit(X, y)
                elif hasattr(model, 'fine_tune'):
                    result = model.fine_tune(y)  # For LSTM, pass target data
                else:
                    # Fallback: retrain model
                    model.fit(X, y)
                    result = {'status': 'success', 'metrics': {'rmse': 0.0, 'mae': 0.0, 'mape': 0.0}}
                
                results[model_name] = result
                
                # Update performance history
                if result['status'] == 'success' and 'metrics' in result:
                    self.performance_history[model_name].append(result['metrics']['rmse'])
                    
                    # Keep only recent performance
                    if len(self.performance_history[model_name]) > self.window_size:
                        self.performance_history[model_name] = self.performance_history[model_name][-self.window_size:]
                
            except Exception as e:
                results[model_name] = {'status': 'error', 'message': str(e)}
        
        # Update ensemble weights
        self._update_weights()
        
        # Calculate ensemble metrics after update
        ensemble_metrics = {'rmse': 0.0, 'mae': 0.0, 'mape': 0.0}
        try:
            if len(X) > 0 and self.is_fitted:
                # Use individual samples for metrics calculation
                test_size = min(3, len(X))
                predictions_list = []
                actual_list = []
                
                for i in range(test_size):
                    # Make prediction for single sample
                    single_X = X[i:i+1]  # Keep as 2D array
                    single_y = y[i]
                    
                    pred = self.predict(single_X, horizon=1)
                    if len(pred) > 0:
                        predictions_list.append(pred[0])  # Take first prediction
                        actual_list.append(single_y)
                
                if len(predictions_list) > 0 and len(predictions_list) == len(actual_list):
                    ensemble_metrics = self._calculate_ensemble_metrics(
                        np.array(actual_list), 
                        np.array(predictions_list)
                    )
                    print(f"Ensemble update metrics: RMSE={ensemble_metrics['rmse']:.4f}, MAE={ensemble_metrics['mae']:.4f}, MAPE={ensemble_metrics['mape']:.2f}%")
        except Exception as e:
            print(f"Warning: Could not calculate ensemble metrics: {e}")
            import traceback
            traceback.print_exc()
        
        return {
            'status': 'success',
            'model_results': results,
            'updated_weights': self.weights.tolist(),
            'metrics': ensemble_metrics
        }
    
    def predict(self, X: np.ndarray, horizon: int = 1) -> np.ndarray:
        """Make ensemble predictions"""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted yet")
        
        predictions = []
        
        for i, model in enumerate(self.models):
            try:
                if hasattr(model, 'predict'):
                    if 'horizon' in model.predict.__code__.co_varnames:
                        pred = model.predict(X, horizon=horizon)
                    else:
                        pred = model.predict(X)
                    
                    # Ensure prediction has the right length
                    if isinstance(pred, np.ndarray):
                        if len(pred) < horizon:
                            # Extend prediction to match horizon
                            last_val = pred[-1] if len(pred) > 0 else 0.0
                            extension = np.full(horizon - len(pred), last_val)
                            pred = np.concatenate([pred, extension])
                        elif len(pred) > horizon:
                            # Truncate to horizon
                            pred = pred[:horizon]
                    else:
                        # Single value prediction, extend to horizon
                        pred = np.full(horizon, pred)
                else:
                    pred = np.zeros(horizon)
                
                predictions.append(pred)
                
            except Exception as e:
                print(f"Error in model {i} prediction: {e}")
                predictions.append(np.zeros(horizon))
        
        # Weighted average of predictions
        predictions = np.array(predictions, dtype=object)  # Use object array to handle different shapes
        
        # Ensure all predictions have the same length (horizon)
        normalized_predictions = []
        
        for i, pred in enumerate(predictions):
            pred_array = np.array(pred)
            
            if len(pred_array) < horizon:
                # Pad with last value
                last_val = pred_array[-1] if len(pred_array) > 0 else 0.0
                padded = np.concatenate([pred_array, np.full(horizon - len(pred_array), last_val)])
                normalized_predictions.append(padded)
            elif len(pred_array) > horizon:
                # Truncate to horizon
                truncated = pred_array[:horizon]
                normalized_predictions.append(truncated)
            else:
                normalized_predictions.append(pred_array)
        
        # Convert to proper numpy array
        normalized_predictions = np.array(normalized_predictions)
        
        # Calculate weighted average
        weighted_pred = np.average(normalized_predictions, axis=0, weights=self.weights)
        
        return weighted_pred
    
    def _update_weights(self):
        """Update model weights based on recent performance"""
        if not self.performance_history:
            return
        
        # Calculate average recent RMSE for each model
        recent_rmse = []
        for model_name in self.model_names:
            history = self.performance_history[model_name]
            if history:
                avg_rmse = np.mean(history[-min(10, len(history)):])  # Last 10 or all available
                recent_rmse.append(avg_rmse)
            else:
                recent_rmse.append(1.0)  # Default high error
        
        recent_rmse = np.array(recent_rmse)
        
        # Convert RMSE to weights (lower RMSE = higher weight)
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        inverse_rmse = 1.0 / (recent_rmse + epsilon)
        
        # Normalize weights
        self.weights = inverse_rmse / np.sum(inverse_rmse)
        
        # Ensure minimum weight for diversity
        min_weight = 0.05
        self.weights = np.maximum(self.weights, min_weight)
        self.weights = self.weights / np.sum(self.weights)
    
    def _calculate_ensemble_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate ensemble performance metrics"""
        try:
            y_true = np.array(y_true, dtype=np.float32)
            y_pred = np.array(y_pred, dtype=np.float32)
            
            rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            mae = float(np.mean(np.abs(y_true - y_pred)))
            
            # Avoid division by zero in MAPE
            nonzero_mask = y_true != 0
            if np.any(nonzero_mask):
                mape = float(np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100)
            else:
                mape = 0.0
                
            return {'rmse': rmse, 'mae': mae, 'mape': mape}
        except Exception as e:
            print(f"Error calculating ensemble metrics: {e}")
            return {'rmse': 0.0, 'mae': 0.0, 'mape': 0.0}
    
    def get_model_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance summary for each model"""
        performance = {}
        
        for i, model_name in enumerate(self.model_names):
            history = self.performance_history[model_name]
            performance[model_name] = {
                'weight': float(self.weights[i]),
                'recent_rmse': float(np.mean(history[-5:])) if len(history) >= 5 else None,
                'avg_rmse': float(np.mean(history)) if history else None,
                'update_count': len(history)
            }
        
        return performance
    
    def save_ensemble(self, filepath: str) -> bool:
        """Save ensemble state"""
        try:
            ensemble_data = {
                'model_names': self.model_names,
                'performance_history': self.performance_history,
                'weights': self.weights.tolist(),
                'window_size': self.window_size,
                'is_fitted': self.is_fitted
            }
            
            joblib.dump(ensemble_data, f"{filepath}_ensemble.pkl")
            
            # Save individual models
            for i, model in enumerate(self.models):
                if hasattr(model, 'save_model'):
                    model.save_model(f"{filepath}_model_{i}")
            
            return True
            
        except Exception as e:
            print(f"Error saving ensemble: {e}")
            return False
    
    def load_ensemble(self, filepath: str) -> bool:
        """Load ensemble state"""
        try:
            ensemble_path = f"{filepath}_ensemble.pkl"
            if not os.path.exists(ensemble_path):
                return False
            
            ensemble_data = joblib.load(ensemble_path)
            self.model_names = ensemble_data['model_names']
            self.performance_history = ensemble_data['performance_history']
            self.weights = np.array(ensemble_data['weights'])
            self.window_size = ensemble_data['window_size']
            self.is_fitted = ensemble_data['is_fitted']
            
            # Load individual models
            for i, model in enumerate(self.models):
                if hasattr(model, 'load_model'):
                    model.load_model(f"{filepath}_model_{i}")
            
            return True
            
        except Exception as e:
            print(f"Error loading ensemble: {e}")
            return False