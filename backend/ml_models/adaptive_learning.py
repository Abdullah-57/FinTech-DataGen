"""
Adaptive Learning Manager for Financial Forecasting

This module manages model versioning, performance tracking, and adaptive updates
for the online learning system. It provides a high-level interface for managing
multiple model versions and automatic rollback capabilities.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import joblib
from .online_learning import OnlineSGDRegressor, OnlineLSTM, AdaptiveEnsemble


class ModelVersion:
    """Represents a specific version of a trained model"""
    
    def __init__(self, version: int, model: Any, metrics: Dict[str, float], 
                 timestamp: datetime, notes: str = ""):
        self.version = version
        self.model = model
        self.metrics = metrics
        self.timestamp = timestamp
        self.notes = notes
        self.is_active = False
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'version': self.version,
            'metrics': self.metrics,
            'timestamp': self.timestamp.isoformat(),
            'notes': self.notes,
            'is_active': self.is_active
        }


class AdaptiveLearningManager:
    """
    Manages adaptive learning for financial forecasting models.
    Handles model versioning, performance tracking, and automatic rollback.
    """
    
    def __init__(self, symbol: str, model_type: str, base_path: str = "backend/ml_models/adaptive_models"):
        self.symbol = symbol
        self.model_type = model_type
        self.base_path = base_path
        self.model_path = os.path.join(base_path, symbol, model_type)
        
        # Model management
        self.versions: List[ModelVersion] = []
        self.current_version = 0
        self.active_model = None
        
        # Performance tracking
        self.performance_threshold = 0.05  # 5% improvement threshold
        self.max_versions = 10  # Keep last 10 versions
        self.rollback_threshold = 0.20  # Rollback if performance degrades by 20%
        
        # Database integration
        self.db = None
        
        # Initialize
        self._ensure_directories()
        self._load_versions()
    
    def set_database(self, db):
        """Set database connection for persistence"""
        self.db = db
    
    def _ensure_directories(self):
        """Create necessary directories"""
        os.makedirs(self.model_path, exist_ok=True)
    
    def _get_version_path(self, version: int) -> str:
        """Get file path for a specific version"""
        return os.path.join(self.model_path, f"v{version}")
    
    def _get_metadata_path(self) -> str:
        """Get metadata file path"""
        return os.path.join(self.model_path, "metadata.json")
    
    def _load_versions(self):
        """Load existing model versions"""
        try:
            metadata_path = self._get_metadata_path()
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.current_version = metadata.get('current_version', 0)
                
                # Load version information
                for version_info in metadata.get('versions', []):
                    version = ModelVersion(
                        version=version_info['version'],
                        model=None,  # Model will be loaded on demand
                        metrics=version_info['metrics'],
                        timestamp=datetime.fromisoformat(version_info['timestamp']),
                        notes=version_info.get('notes', '')
                    )
                    version.is_active = version_info.get('is_active', False)
                    self.versions.append(version)
                
                # Load active model
                if self.current_version > 0:
                    self._load_model(self.current_version)
                    
        except Exception as e:
            print(f"Error loading versions: {e}")
            self.versions = []
            self.current_version = 0
    
    def _save_metadata(self):
        """Save version metadata"""
        try:
            metadata = {
                'symbol': self.symbol,
                'model_type': self.model_type,
                'current_version': self.current_version,
                'versions': [v.to_dict() for v in self.versions],
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self._get_metadata_path(), 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            print(f"Error saving metadata: {e}")
    
    def _create_model(self) -> Any:
        """Create a new model instance based on model type"""
        if self.model_type == 'sgd':
            return OnlineSGDRegressor(learning_rate=0.01, alpha=0.0001)
        elif self.model_type == 'lstm':
            return OnlineLSTM(lookback=10, lstm_units=32, learning_rate=0.001)
        elif self.model_type == 'ensemble':
            # Create ensemble with SGD and LSTM
            sgd_model = OnlineSGDRegressor(learning_rate=0.01, alpha=0.0001)
            lstm_model = OnlineLSTM(lookback=10, lstm_units=32, learning_rate=0.001)
            return AdaptiveEnsemble([sgd_model, lstm_model], window_size=50)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _load_model(self, version: int) -> bool:
        """Load a specific model version"""
        try:
            version_path = self._get_version_path(version)
            
            # Create model instance
            model = self._create_model()
            
            # Load model state
            if hasattr(model, 'load_model'):
                success = model.load_model(version_path)
            elif hasattr(model, 'load_ensemble'):
                success = model.load_ensemble(version_path)
            else:
                success = False
            
            if success:
                self.active_model = model
                return True
            else:
                print(f"Failed to load model version {version}")
                return False
                
        except Exception as e:
            print(f"Error loading model version {version}: {e}")
            return False
    
    def _save_model(self, model: Any, version: int) -> bool:
        """Save a model version"""
        try:
            version_path = self._get_version_path(version)
            
            if hasattr(model, 'save_model'):
                return model.save_model(version_path)
            elif hasattr(model, 'save_ensemble'):
                return model.save_ensemble(version_path)
            else:
                return False
                
        except Exception as e:
            print(f"Error saving model version {version}: {e}")
            return False
    
    def initial_training(self, X: np.ndarray, y: np.ndarray, notes: str = "") -> Dict[str, Any]:
        """Perform initial training and create first version"""
        try:
            # Create new model
            model = self._create_model()
            
            # Train model
            if hasattr(model, 'initial_fit'):
                # Handle different model types
                if self.model_type == 'lstm':
                    # LSTM expects time series data (y values only)
                    result = model.initial_fit(y, epochs=20, batch_size=8)
                elif self.model_type == 'ensemble':
                    # Ensemble expects both X and y
                    result = model.initial_fit(X, y)
                    # Ensure ensemble result has metrics
                    if 'metrics' not in result or not result['metrics']:
                        result['metrics'] = {'rmse': 0.0, 'mae': 0.0, 'mape': 0.0}
                else:
                    # SGD and other models expect X and y
                    result = model.initial_fit(X, y)
            else:
                # Fallback for models without initial_fit
                model.fit(X, y)
                result = {'status': 'success', 'metrics': {'rmse': 0.0, 'mae': 0.0, 'mape': 0.0}}
            
            if result['status'] != 'success':
                return result
            
            # Create first version
            version = 1
            model_version = ModelVersion(
                version=version,
                model=model,
                metrics=result.get('metrics', {}),
                timestamp=datetime.now(),
                notes=notes or "Initial training"
            )
            model_version.is_active = True
            
            # Save model
            if self._save_model(model, version):
                self.versions = [model_version]
                self.current_version = version
                self.active_model = model
                self._save_metadata()
                
                # Save to database
                self._save_to_database(model_version, result)
                
                return {
                    'status': 'success',
                    'version': version,
                    'metrics': result.get('metrics', {}),
                    'message': 'Initial training completed successfully'
                }
            else:
                return {'status': 'error', 'message': 'Failed to save model'}
                
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def update_model(self, X: np.ndarray, y: np.ndarray, force_new_version: bool = False) -> Dict[str, Any]:
        """Update model with new data and potentially create new version"""
        try:
            if self.active_model is None:
                return self.initial_training(X, y, "Auto-initialized from update")
            
            # Update current model
            if hasattr(self.active_model, 'partial_fit'):
                result = self.active_model.partial_fit(X, y)
            elif hasattr(self.active_model, 'fine_tune'):
                # LSTM fine-tuning expects time series data (y values only)
                result = self.active_model.fine_tune(y, epochs=3)
            elif hasattr(self.active_model, 'update'):
                result = self.active_model.update(X, y)
                # Ensure ensemble update result has metrics
                if self.model_type == 'ensemble' and ('metrics' not in result or not result['metrics']):
                    result['metrics'] = {'rmse': 0.0, 'mae': 0.0, 'mape': 0.0}
            else:
                return {'status': 'error', 'message': 'Model does not support incremental updates'}
            
            if result['status'] != 'success':
                return result
            
            current_metrics = result.get('metrics', {})
            
            # Check if we should create a new version
            should_create_version = force_new_version or self._should_create_version(current_metrics)
            
            if should_create_version:
                return self._create_new_version(current_metrics, "Performance improvement detected")
            else:
                # Just update the current version's performance
                if self.versions:
                    self.versions[-1].metrics = current_metrics
                    self._save_metadata()
                
                return {
                    'status': 'success',
                    'version': self.current_version,
                    'metrics': current_metrics,
                    'message': 'Model updated successfully',
                    'new_version_created': False
                }
                
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _should_create_version(self, new_metrics: Dict[str, float]) -> bool:
        """Determine if a new version should be created based on performance"""
        if not self.versions:
            return True
        
        current_version = self.versions[-1]
        current_rmse = current_version.metrics.get('rmse', float('inf'))
        new_rmse = new_metrics.get('rmse', float('inf'))
        
        # Create new version if RMSE improved by threshold percentage
        improvement = (current_rmse - new_rmse) / current_rmse if current_rmse > 0 else 0
        return improvement >= self.performance_threshold
    
    def _create_new_version(self, metrics: Dict[str, float], notes: str = "") -> Dict[str, Any]:
        """Create a new model version"""
        try:
            # Deactivate current version
            for version in self.versions:
                version.is_active = False
            
            # Create new version
            new_version_num = self.current_version + 1
            new_version = ModelVersion(
                version=new_version_num,
                model=self.active_model,
                metrics=metrics,
                timestamp=datetime.now(),
                notes=notes
            )
            new_version.is_active = True
            
            # Save model
            if self._save_model(self.active_model, new_version_num):
                self.versions.append(new_version)
                self.current_version = new_version_num
                
                # Cleanup old versions
                self._cleanup_old_versions()
                
                # Save metadata
                self._save_metadata()
                
                # Save to database
                self._save_to_database(new_version, {'metrics': metrics})
                
                return {
                    'status': 'success',
                    'version': new_version_num,
                    'metrics': metrics,
                    'message': f'New version {new_version_num} created successfully',
                    'new_version_created': True
                }
            else:
                return {'status': 'error', 'message': 'Failed to save new version'}
                
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _cleanup_old_versions(self):
        """Remove old versions beyond the maximum limit"""
        if len(self.versions) > self.max_versions:
            # Keep the most recent versions
            versions_to_remove = self.versions[:-self.max_versions]
            self.versions = self.versions[-self.max_versions:]
            
            # Delete old version files
            for version in versions_to_remove:
                try:
                    version_path = self._get_version_path(version.version)
                    if os.path.exists(version_path):
                        import shutil
                        shutil.rmtree(version_path, ignore_errors=True)
                except Exception as e:
                    print(f"Error removing old version {version.version}: {e}")
    
    def rollback_to_version(self, version: int) -> Dict[str, Any]:
        """Rollback to a specific version"""
        try:
            # Find the version
            target_version = None
            for v in self.versions:
                if v.version == version:
                    target_version = v
                    break
            
            if target_version is None:
                return {'status': 'error', 'message': f'Version {version} not found'}
            
            # Load the model
            if self._load_model(version):
                # Update active version
                for v in self.versions:
                    v.is_active = False
                target_version.is_active = True
                self.current_version = version
                
                self._save_metadata()
                
                return {
                    'status': 'success',
                    'version': version,
                    'metrics': target_version.metrics,
                    'message': f'Successfully rolled back to version {version}'
                }
            else:
                return {'status': 'error', 'message': f'Failed to load version {version}'}
                
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def auto_rollback_check(self, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Check if automatic rollback is needed based on performance degradation"""
        if len(self.versions) < 2:
            return {'rollback_needed': False}
        
        # Compare with previous version
        previous_version = self.versions[-2]
        previous_rmse = previous_version.metrics.get('rmse', 0)
        current_rmse = current_metrics.get('rmse', float('inf'))
        
        if previous_rmse > 0:
            degradation = (current_rmse - previous_rmse) / previous_rmse
            
            if degradation >= self.rollback_threshold:
                # Perform automatic rollback
                rollback_result = self.rollback_to_version(previous_version.version)
                
                return {
                    'rollback_needed': True,
                    'rollback_performed': rollback_result['status'] == 'success',
                    'degradation_percent': degradation * 100,
                    'rollback_result': rollback_result
                }
        
        return {'rollback_needed': False}
    
    def predict(self, X: np.ndarray, horizon: int = 1) -> np.ndarray:
        """Make predictions using the active model"""
        if self.active_model is None:
            raise ValueError("No active model available for prediction")
        
        if hasattr(self.active_model, 'predict'):
            if self.model_type == 'lstm':
                # LSTM predict method expects data and horizon
                if 'horizon' in self.active_model.predict.__code__.co_varnames:
                    return self.active_model.predict(X, horizon=horizon)
                else:
                    return self.active_model.predict(X)
            else:
                # SGD and other models
                if 'horizon' in self.active_model.predict.__code__.co_varnames:
                    return self.active_model.predict(X, horizon=horizon)
                else:
                    return self.active_model.predict(X)
        else:
            raise ValueError("Active model does not support prediction")
    
    def get_version_history(self) -> List[Dict[str, Any]]:
        """Get history of all model versions"""
        return [v.to_dict() for v in self.versions]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all versions"""
        if not self.versions:
            return {'message': 'No versions available'}
        
        # Calculate performance trends
        rmse_history = [v.metrics.get('rmse', 0) for v in self.versions]
        mae_history = [v.metrics.get('mae', 0) for v in self.versions]
        
        return {
            'total_versions': len(self.versions),
            'current_version': self.current_version,
            'best_rmse': min(rmse_history) if rmse_history else 0,
            'latest_rmse': rmse_history[-1] if rmse_history else 0,
            'rmse_trend': rmse_history,
            'mae_trend': mae_history,
            'version_timestamps': [v.timestamp.isoformat() for v in self.versions]
        }
    
    def _save_to_database(self, version: ModelVersion, training_result: Dict[str, Any]):
        """Save version information to database"""
        if self.db is None:
            return
        
        try:
            # Save model version
            version_doc = {
                'symbol': self.symbol,
                'model_type': self.model_type,
                'version': version.version,
                'created_at': version.timestamp,
                'performance_metrics': version.metrics,
                'file_path': self._get_version_path(version.version),
                'is_active': version.is_active,
                'training_samples': training_result.get('samples_trained', 0),
                'notes': version.notes
            }
            
            # Use the adaptive learning collections
            if hasattr(self.db, 'db') and self.db.db is not None:
                self.db.db.model_versions.insert_one(version_doc)
            
            # Save training event
            training_event = {
                'symbol': self.symbol,
                'model_type': self.model_type,
                'trigger_type': 'manual',  # Could be 'scheduled' or 'automatic'
                'status': 'completed',
                'timestamp': datetime.now(),
                'version_created': version.version,
                'performance_metrics': version.metrics,
                'training_samples': training_result.get('samples_trained', 0)
            }
            
            if hasattr(self.db, 'db') and self.db.db is not None:
                self.db.db.training_events.insert_one(training_event)
                
        except Exception as e:
            print(f"Error saving to database: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics from database"""
        if self.db is None or not hasattr(self.db, 'db') or self.db.db is None:
            return {'error': 'Database not available'}
        
        try:
            stats = {}
            
            # Model versions count
            stats['total_versions'] = self.db.db.model_versions.count_documents({
                'symbol': self.symbol,
                'model_type': self.model_type
            })
            
            # Training events count
            stats['total_training_events'] = self.db.db.training_events.count_documents({
                'symbol': self.symbol,
                'model_type': self.model_type
            })
            
            # Latest performance
            latest_version = self.db.db.model_versions.find_one(
                {'symbol': self.symbol, 'model_type': self.model_type},
                sort=[('version', -1)]
            )
            
            if latest_version:
                stats['latest_performance'] = latest_version.get('performance_metrics', {})
                stats['latest_version'] = latest_version.get('version', 0)
            
            return stats
            
        except Exception as e:
            return {'error': str(e)}