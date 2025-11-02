"""
Continuous Learning Manager for Financial Forecasting

This module orchestrates the continuous learning process, managing data fetching,
preprocessing, model updates, and scheduling for the adaptive learning system.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from .adaptive_learning import AdaptiveLearningManager
import schedule
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContinuousLearningManager:
    """
    Orchestrates continuous learning for financial forecasting models.
    Manages automatic data fetching, preprocessing, and model updates.
    """
    
    def __init__(self, db=None):
        self.db = db
        self.adaptive_managers: Dict[str, AdaptiveLearningManager] = {}
        self.is_running = False
        self.scheduler_thread = None
        
        # Configuration
        self.update_schedules = {
            'sgd': {'hours': 6},      # SGD updates every 6 hours
            'lstm': {'hours': 24},    # LSTM updates daily
            'ensemble': {'hours': 12} # Ensemble updates every 12 hours
        }
        
        # Data preprocessing settings
        self.min_samples_for_update = 10
        self.lookback_days = 30
        
        # Performance monitoring
        self.performance_alerts = []
        self.last_update_times = {}
        
    def register_symbol(self, symbol: str, model_types: List[str]) -> Dict[str, Any]:
        """Register a symbol for continuous learning"""
        try:
            results = {}
            
            for model_type in model_types:
                if model_type not in self.update_schedules:
                    results[model_type] = {'status': 'error', 'message': f'Unknown model type: {model_type}'}
                    continue
                
                # Create adaptive learning manager
                manager_key = f"{symbol}_{model_type}"
                manager = AdaptiveLearningManager(symbol, model_type)
                manager.set_database(self.db)
                
                self.adaptive_managers[manager_key] = manager
                
                # Schedule updates
                self._schedule_updates(symbol, model_type)
                
                results[model_type] = {'status': 'success', 'message': 'Registered successfully'}
                
                logger.info(f"Registered {symbol} for {model_type} continuous learning")
            
            return {
                'status': 'success',
                'symbol': symbol,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Error registering symbol {symbol}: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _schedule_updates(self, symbol: str, model_type: str):
        """Schedule automatic updates for a symbol and model type"""
        schedule_config = self.update_schedules[model_type]
        
        if 'hours' in schedule_config:
            schedule.every(schedule_config['hours']).hours.do(
                self._scheduled_update, symbol, model_type
            )
        elif 'minutes' in schedule_config:
            schedule.every(schedule_config['minutes']).minutes.do(
                self._scheduled_update, symbol, model_type
            )
    
    def _scheduled_update(self, symbol: str, model_type: str):
        """Perform a scheduled model update"""
        try:
            logger.info(f"Starting scheduled update for {symbol} {model_type}")
            
            # Fetch new data
            data = self._fetch_recent_data(symbol)
            
            if data is None or len(data) < self.min_samples_for_update:
                logger.warning(f"Insufficient data for {symbol} update: {len(data) if data is not None else 0} samples")
                return
            
            # Preprocess data
            X, y = self._preprocess_data(data)
            
            if len(X) == 0:
                logger.warning(f"No valid samples after preprocessing for {symbol}")
                return
            
            # Update model
            manager_key = f"{symbol}_{model_type}"
            if manager_key in self.adaptive_managers:
                manager = self.adaptive_managers[manager_key]
                result = manager.update_model(X, y)
                
                # Log result
                if result['status'] == 'success':
                    logger.info(f"Successfully updated {symbol} {model_type}: version {result['version']}")
                    
                    # Check for performance alerts
                    self._check_performance_alerts(symbol, model_type, result['metrics'])
                    
                    # Update last update time
                    self.last_update_times[manager_key] = datetime.now()
                else:
                    logger.error(f"Failed to update {symbol} {model_type}: {result['message']}")
            
        except Exception as e:
            logger.error(f"Error in scheduled update for {symbol} {model_type}: {e}")
    
    def _fetch_recent_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch recent data for a symbol"""
        try:
            if self.db is None:
                logger.warning("No database connection available")
                return None
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days)
            
            # Fetch historical prices
            prices = self.db.get_prices(
                symbol=symbol,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                limit=1000
            )
            
            if not prices:
                logger.warning(f"No price data found for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(prices)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _preprocess_data(self, df: pd.DataFrame) -> tuple:
        """Preprocess data for model training"""
        try:
            # Calculate technical indicators
            df = self._calculate_technical_indicators(df)
            
            # Create features and targets
            feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'returns', 'volatility', 'sma_5', 'sma_20', 'rsi'
            ]
            
            # Ensure all feature columns exist
            for col in feature_columns:
                if col not in df.columns:
                    df[col] = 0.0
            
            # Remove rows with NaN values
            df = df.dropna()
            
            if len(df) < 2:
                return np.array([]), np.array([])
            
            # Features (all rows except last)
            X = df[feature_columns].iloc[:-1].values
            
            # Targets (next day's close price)
            y = df['close'].iloc[1:].values
            
            return X.astype(np.float32), y.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            return np.array([]), np.array([])
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for features"""
        try:
            # Returns
            df['returns'] = df['close'].pct_change()
            
            # Volatility (rolling standard deviation of returns)
            df['volatility'] = df['returns'].rolling(window=20).std()
            
            # Simple Moving Averages
            df['sma_5'] = df['close'].rolling(window=5).mean()
            df['sma_20'] = df['close'].rolling(window=20).mean()
            
            # RSI (Relative Strength Index)
            df['rsi'] = self._calculate_rsi(df['close'])
            
            # Fill NaN values with forward fill then backward fill
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return pd.Series([50] * len(prices), index=prices.index)
    
    def _check_performance_alerts(self, symbol: str, model_type: str, metrics: Dict[str, float]):
        """Check for performance degradation and create alerts"""
        try:
            manager_key = f"{symbol}_{model_type}"
            manager = self.adaptive_managers.get(manager_key)
            
            if manager is None:
                return
            
            # Check for automatic rollback
            rollback_check = manager.auto_rollback_check(metrics)
            
            if rollback_check.get('rollback_needed', False):
                alert = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'model_type': model_type,
                    'type': 'performance_degradation',
                    'message': f"Performance degraded by {rollback_check.get('degradation_percent', 0):.2f}%",
                    'rollback_performed': rollback_check.get('rollback_performed', False)
                }
                
                self.performance_alerts.append(alert)
                logger.warning(f"Performance alert: {alert['message']}")
            
        except Exception as e:
            logger.error(f"Error checking performance alerts: {e}")
    
    def manual_update(self, symbol: str, model_type: str) -> Dict[str, Any]:
        """Manually trigger a model update"""
        try:
            manager_key = f"{symbol}_{model_type}"
            
            if manager_key not in self.adaptive_managers:
                return {'status': 'error', 'message': f'No manager found for {symbol} {model_type}'}
            
            # Fetch and preprocess data
            data = self._fetch_recent_data(symbol)
            
            if data is None or len(data) < self.min_samples_for_update:
                return {
                    'status': 'error', 
                    'message': f'Insufficient data: {len(data) if data is not None else 0} samples'
                }
            
            X, y = self._preprocess_data(data)
            
            if len(X) == 0:
                return {'status': 'error', 'message': 'No valid samples after preprocessing'}
            
            # Update model
            manager = self.adaptive_managers[manager_key]
            result = manager.update_model(X, y)
            
            # Update last update time
            if result['status'] == 'success':
                self.last_update_times[manager_key] = datetime.now()
            
            return result
            
        except Exception as e:
            logger.error(f"Error in manual update for {symbol} {model_type}: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def initial_training(self, symbol: str, model_type: str) -> Dict[str, Any]:
        """Perform initial training for a model"""
        try:
            manager_key = f"{symbol}_{model_type}"
            
            if manager_key not in self.adaptive_managers:
                return {'status': 'error', 'message': f'No manager found for {symbol} {model_type}'}
            
            # Fetch historical data (more data for initial training)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)  # 3 months for initial training
            
            prices = self.db.get_prices(
                symbol=symbol,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                limit=2000
            )
            
            if not prices or len(prices) < 50:
                return {
                    'status': 'error', 
                    'message': f'Insufficient historical data: {len(prices) if prices else 0} samples'
                }
            
            # Preprocess data
            df = pd.DataFrame(prices)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            X, y = self._preprocess_data(df)
            
            if len(X) < 20:
                return {'status': 'error', 'message': 'Insufficient valid samples for initial training'}
            
            # Perform initial training
            manager = self.adaptive_managers[manager_key]
            result = manager.initial_training(X, y, f"Initial training with {len(X)} samples")
            
            # Update last update time
            if result['status'] == 'success':
                self.last_update_times[manager_key] = datetime.now()
            
            return result
            
        except Exception as e:
            logger.error(f"Error in initial training for {symbol} {model_type}: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def predict(self, symbol: str, model_type: str, horizon: int = 1) -> Dict[str, Any]:
        """Make predictions using a trained model"""
        try:
            manager_key = f"{symbol}_{model_type}"
            
            if manager_key not in self.adaptive_managers:
                return {'status': 'error', 'message': f'No manager found for {symbol} {model_type}'}
            
            # Fetch recent data for prediction
            data = self._fetch_recent_data(symbol)
            
            if data is None or len(data) == 0:
                return {'status': 'error', 'message': 'No data available for prediction'}
            
            # Preprocess data
            X, _ = self._preprocess_data(data)
            
            if len(X) == 0:
                return {'status': 'error', 'message': 'No valid samples for prediction'}
            
            # Make prediction
            manager = self.adaptive_managers[manager_key]
            if model_type == 'lstm':
                # For LSTM, use the last few values from y (close prices) for prediction
                recent_prices = data['close'].iloc[-20:].values  # Use last 20 close prices
                predictions = manager.predict(recent_prices, horizon=horizon)
            else:
                # For SGD and ensemble models, use feature matrix
                predictions = manager.predict(X[-1:], horizon=horizon)
            
            # Debug logging
            logger.info(f"Prediction for {symbol} {model_type}: requested horizon={horizon}, got {len(predictions)} predictions")
            
            # Ensure we have the right number of predictions
            if len(predictions) != horizon:
                logger.warning(f"Prediction length mismatch: expected {horizon}, got {len(predictions)}")
                # Try to fix the mismatch
                if len(predictions) < horizon:
                    # Extend with last value
                    last_val = predictions[-1] if len(predictions) > 0 else 0.0
                    predictions = np.concatenate([predictions, np.full(horizon - len(predictions), last_val)])
                else:
                    # Truncate to horizon
                    predictions = predictions[:horizon]
            
            return {
                'status': 'success',
                'symbol': symbol,
                'model_type': model_type,
                'horizon': horizon,
                'predictions': predictions.tolist(),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error making prediction for {symbol} {model_type}: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def start_continuous_learning(self):
        """Start the continuous learning scheduler"""
        if self.is_running:
            logger.warning("Continuous learning is already running")
            return
        
        self.is_running = True
        
        def run_scheduler():
            logger.info("Starting continuous learning scheduler")
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            logger.info("Continuous learning scheduler stopped")
        
        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Continuous learning started")
    
    def stop_continuous_learning(self):
        """Stop the continuous learning scheduler"""
        self.is_running = False
        
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
        
        logger.info("Continuous learning stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of continuous learning system"""
        status = {
            'is_running': self.is_running,
            'registered_models': len(self.adaptive_managers),
            'models': {},
            'recent_alerts': self.performance_alerts[-10:],  # Last 10 alerts
            'last_update_times': {}
        }
        
        # Get status for each registered model
        for manager_key, manager in self.adaptive_managers.items():
            symbol, model_type = manager_key.split('_', 1)
            
            performance_summary = manager.get_performance_summary()
            
            status['models'][manager_key] = {
                'symbol': symbol,
                'model_type': model_type,
                'current_version': performance_summary.get('current_version', 0),
                'total_versions': performance_summary.get('total_versions', 0),
                'latest_rmse': performance_summary.get('latest_rmse', 0),
                'best_rmse': performance_summary.get('best_rmse', 0)
            }
            
            # Last update time
            if manager_key in self.last_update_times:
                status['last_update_times'][manager_key] = self.last_update_times[manager_key].isoformat()
        
        return status
    
    def get_model_performance(self, symbol: str, model_type: str) -> Dict[str, Any]:
        """Get detailed performance information for a specific model"""
        manager_key = f"{symbol}_{model_type}"
        
        if manager_key not in self.adaptive_managers:
            return {'status': 'error', 'message': f'No manager found for {symbol} {model_type}'}
        
        manager = self.adaptive_managers[manager_key]
        
        return {
            'status': 'success',
            'symbol': symbol,
            'model_type': model_type,
            'performance_summary': manager.get_performance_summary(),
            'version_history': manager.get_version_history(),
            'database_stats': manager.get_database_stats()
        }
    
    def rollback_model(self, symbol: str, model_type: str, version: int) -> Dict[str, Any]:
        """Rollback a model to a specific version"""
        manager_key = f"{symbol}_{model_type}"
        
        if manager_key not in self.adaptive_managers:
            return {'status': 'error', 'message': f'No manager found for {symbol} {model_type}'}
        
        manager = self.adaptive_managers[manager_key]
        return manager.rollback_to_version(version)
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Cleanup old training events and performance data"""
        try:
            if self.db is None or not hasattr(self.db, 'db') or self.db.db is None:
                return {'status': 'error', 'message': 'Database not available'}
            
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Cleanup old training events
            result1 = self.db.db.training_events.delete_many({
                'timestamp': {'$lt': cutoff_date}
            })
            
            # Cleanup old performance records (if they exist)
            result2 = self.db.db.performance_history.delete_many({
                'timestamp': {'$lt': cutoff_date}
            }) if 'performance_history' in self.db.db.list_collection_names() else None
            
            return {
                'status': 'success',
                'training_events_deleted': result1.deleted_count,
                'performance_records_deleted': result2.deleted_count if result2 else 0
            }
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return {'status': 'error', 'message': str(e)}