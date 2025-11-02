from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

class MongoDB:
    def __init__(self):
        self.client = None
        self.db = None
        self.connect()
        # Lazily cached collection handles
        self._datasets_col = None
        self._predictions_col = None
        self._historical_col = None
        self._metadata_col = None
    
    def connect(self):
        """Connect to MongoDB"""
        try:
            mongodb_uri = os.getenv('MONGOURI')
            if not mongodb_uri:
                raise ValueError("MONGOURI environment variable not set")
            
            self.client = MongoClient(mongodb_uri)
            self.db = self.client.fintech
            # Initialize common collections
            self._datasets_col = self.db.datasets
            self._predictions_col = self.db.predictions
            self._historical_col = self.db.historical_prices
            self._metadata_col = self.db.metadata
            
            # Test connection
            self.client.admin.command('ping')
            print("✅ Connected to MongoDB successfully")
            
        except Exception as e:
            print(f"❌ Failed to connect to MongoDB: {e}")
            self.client = None
            self.db = None
    
    def test_connection(self):
        """Test database connection"""
        try:
            if self.client:
                self.client.admin.command('ping')
                return True
        except:
            pass
        return False
    
    def save_dataset(self, dataset_data):
        """Save dataset to database"""
        try:
            if self.db is None:
                raise Exception("Database not connected")
            
            collection = self._datasets_col or self.db.datasets
            result = collection.insert_one(dataset_data)
            return result
        except Exception as e:
            print(f"Error saving dataset: {e}")
            raise
    
    def get_dataset_by_id(self, dataset_id):
        """Get dataset by ID"""
        try:
            if self.db is None:
                raise Exception("Database not connected")
            
            collection = self._datasets_col or self.db.datasets
            from bson import ObjectId
            return collection.find_one({"_id": ObjectId(dataset_id)})
        except Exception as e:
            print(f"Error getting dataset: {e}")
            return None
    
    def get_all_datasets(self):
        """Get all datasets"""
        try:
            if self.db is None:
                raise Exception("Database not connected")
            
            collection = self._datasets_col or self.db.datasets
            datasets = list(collection.find().sort("generated_at", -1))
            
            # Convert ObjectId to string and handle NaN values for JSON serialization
            for dataset in datasets:
                dataset['_id'] = str(dataset['_id'])
                # Clean up data array to handle NaN values
                if 'data' in dataset and isinstance(dataset['data'], list):
                    for record in dataset['data']:
                        self._clean_nan_values(record)
            
            return datasets
        except Exception as e:
            print(f"Error getting datasets: {e}")
            return []
    
    def _clean_nan_values(self, obj):
        """Recursively clean NaN values from a dictionary, converting them to None"""
        import math
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, float) and math.isnan(value):
                    obj[key] = None
                elif isinstance(value, dict):
                    self._clean_nan_values(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            self._clean_nan_values(item)
        return obj
    
    def get_recent_datasets(self, limit=10):
        """Get recent datasets"""
        try:
            if self.db is None:
                raise Exception("Database not connected")
            
            collection = self._datasets_col or self.db.datasets
            datasets = list(collection.find().sort("generated_at", -1).limit(limit))
            
            # Convert ObjectId to string and format for frontend
            formatted_datasets = []
            for dataset in datasets:
                formatted_datasets.append({
                    'id': str(dataset['_id']),
                    'symbol': dataset.get('symbol', 'N/A'),
                    'date': dataset.get('generated_at', datetime.now()).strftime('%Y-%m-%d'),
                    'records': dataset.get('records', 0),
                    'exchange': dataset.get('exchange', 'N/A')
                })
            
            return formatted_datasets
        except Exception as e:
            print(f"Error getting recent datasets: {e}")
            return []
    
    def get_latest_data(self, symbol):
        """Get latest data for a symbol"""
        try:
            if self.db is None:
                raise Exception("Database not connected")
            
            collection = self._datasets_col or self.db.datasets
            latest = collection.find_one(
                {"symbol": symbol},
                sort=[("generated_at", -1)]
            )
            return latest
        except Exception as e:
            print(f"Error getting latest data: {e}")
            return None
    
    def save_prediction(self, prediction_data):
        """Save prediction to database"""
        try:
            if self.db is None:
                raise Exception("Database not connected")
            
            collection = self._predictions_col or self.db.predictions
            result = collection.insert_one(prediction_data)
            return result
        except Exception as e:
            print(f"Error saving prediction: {e}")
            raise
    
    def get_recent_predictions(self, limit=10):
        """Get recent predictions"""
        try:
            if self.db is None:
                raise Exception("Database not connected")
            
            collection = self._predictions_col or self.db.predictions
            predictions = list(collection.find().sort("created_at", -1).limit(limit))
            
            # Convert ObjectId to string
            for prediction in predictions:
                prediction['_id'] = str(prediction['_id'])
            
            return predictions
        except Exception as e:
            print(f"Error getting predictions: {e}")
            return []

    # New: Historical Prices APIs
    def save_historical_prices(self, symbol, exchange, prices):
        """Bulk insert curated OHLCV rows into historical_prices.
        prices: list of dicts with keys date, open, high, low, close, volume
        """
        try:
            if self.db is None:
                raise Exception("Database not connected")
            collection = self._historical_col or self.db.historical_prices
            if not prices:
                print("⚠️ No prices data provided to save_historical_prices")
                return None
            
            print(f"💾 Saving {len(prices)} historical price records for {symbol}")
            
            # First, remove existing records for this symbol to avoid duplicates
            collection.delete_many({'symbol': symbol, 'exchange': exchange})
            
            docs = []
            for i, p in enumerate(prices):
                try:
                    doc = {
                        'symbol': symbol,
                        'exchange': exchange,
                        'date': p.get('date'),
                        'open': float(p.get('open_price') or p.get('open') or 0),
                        'high': float(p.get('high_price') or p.get('high') or 0),
                        'low': float(p.get('low_price') or p.get('low') or 0),
                        'close': float(p.get('close_price') or p.get('close') or 0),
                        'volume': int(p.get('volume') or 0)
                    }
                    docs.append(doc)
                except Exception as e:
                    print(f"⚠️ Error processing price record {i}: {e}")
                    continue
            
            if docs:
                result = collection.insert_many(docs, ordered=False)
                print(f"✅ Successfully saved {len(result.inserted_ids)} historical price records")
                return result
            else:
                print("❌ No valid price records to save")
                return None
        except Exception as e:
            print(f"❌ Error saving historical prices: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_prices(self, symbol, start_date=None, end_date=None, limit=500):
        """Query historical OHLCV for visualization."""
        try:
            if self.db is None:
                raise Exception("Database not connected")
            collection = self._historical_col or self.db.historical_prices
            query = {'symbol': symbol}
            if start_date or end_date:
                query['date'] = {}
                if start_date:
                    query['date']['$gte'] = start_date
                if end_date:
                    query['date']['$lte'] = end_date
            cursor = collection.find(query).sort('date', 1).limit(int(limit) if limit else 0)
            rows = []
            for doc in cursor:
                rows.append({
                    'symbol': doc.get('symbol'),
                    'date': doc.get('date'),
                    'open': float(doc.get('open', 0)),
                    'high': float(doc.get('high', 0)),
                    'low': float(doc.get('low', 0)),
                    'close': float(doc.get('close', 0)),
                    'volume': int(doc.get('volume', 0))
                })
            return rows
        except Exception as e:
            print(f"Error getting prices: {e}")
            return []

    # New: Forecast and Metadata helpers
    def save_forecast(self, forecast_data):
        """Insert a new forecast document with model, timestamp, horizon, predicted_values."""
        try:
            if self.db is None:
                raise Exception("Database not connected")
            collection = self._predictions_col or self.db.predictions
            result = collection.insert_one(forecast_data)
            return result
        except Exception as e:
            print(f"Error saving forecast: {e}")
            raise

    def get_predictions(self, symbol=None, horizon=None, model=None, limit=50):
        """Query predictions filtered by symbol, horizon, and model."""
        try:
            if self.db is None:
                raise Exception("Database not connected")
            collection = self._predictions_col or self.db.predictions
            query = {}
            if symbol:
                query['symbol'] = symbol
            if horizon:
                query['forecast_horizon'] = horizon
            if model:
                query['model'] = model
            print(f"🔍 Querying predictions with filters: {query}")
            cursor = collection.find(query).sort('created_at', -1).limit(int(limit) if limit else 0)
            results = []
            for doc in cursor:
                doc['_id'] = str(doc['_id'])
                results.append(doc)
            print(f"📊 Found {len(results)} predictions matching query")
            return results
        except Exception as e:
            print(f"Error getting predictions: {e}")
            return []

    def upsert_metadata(self, symbol, metadata):
        """Upsert instrument metadata: instrument info, data sources, update logs."""
        try:
            if self.db is None:
                raise Exception("Database not connected")
            
            print(f"💾 Upserting metadata for symbol: {symbol}")
            collection = self._metadata_col or self.db.metadata
            from pymongo import ReturnDocument
            
            update_doc = {
                '$set': {
                    'symbol': symbol,
                    'instrument_info': metadata.get('instrument_info'),
                    'data_sources': metadata.get('data_sources'),
                    'last_updated': datetime.now()
                },
                '$setOnInsert': {'created_at': datetime.now()}
            }
            
            # Add update logs if provided
            if metadata.get('update_logs'):
                update_doc['$push'] = {'update_logs': {'$each': metadata.get('update_logs', [])}}
            
            updated = collection.find_one_and_update(
                {'symbol': symbol},
                update_doc,
                upsert=True,
                return_document=ReturnDocument.AFTER
            )
            
            if updated and '_id' in updated:
                updated['_id'] = str(updated['_id'])
            
            print(f"✅ Successfully upserted metadata for {symbol}")
            return updated
        except Exception as e:
            print(f"❌ Error upserting metadata for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_metadata(self, symbol=None):
        """Get metadata for a symbol or all instruments."""
        try:
            if self.db is None:
                raise Exception("Database not connected")
            collection = self._metadata_col or self.db.metadata
            if symbol:
                doc = collection.find_one({'symbol': symbol})
                if doc and '_id' in doc:
                    doc['_id'] = str(doc['_id'])
                return doc
            docs = list(collection.find())
            for d in docs:
                d['_id'] = str(d['_id'])
            return docs
        except Exception as e:
            print(f"Error getting metadata: {e}")
            return None if symbol else []
    
    def count_datasets(self):
        """Count total datasets"""
        try:
            if self.db is None:
                return 0
            
            collection = self._datasets_col or self.db.datasets
            return collection.count_documents({})
        except:
            return 0
    
    def count_records(self):
        """Count total records across all datasets"""
        try:
            if self.db is None:
                return 0
            
            collection = self._datasets_col or self.db.datasets
            pipeline = [{"$group": {"_id": None, "total": {"$sum": "$records"}}}]
            result = list(collection.aggregate(pipeline))
            return result[0]['total'] if result else 0
        except:
            return 0
    
    def get_last_generated_date(self):
        """Get last generated date"""
        try:
            if self.db is None:
                return None
            
            collection = self._datasets_col or self.db.datasets
            latest = collection.find_one(sort=[("generated_at", -1)])
            return latest['generated_at'].strftime('%Y-%m-%d %H:%M:%S') if latest else None
        except:
            return None

    # Adaptive Learning Extensions
    def get_adaptive_learning_stats(self):
        """Get adaptive learning statistics from database"""
        try:
            if self.db is None:
                return {'error': 'Database not connected'}
            
            stats = {}
            
            # Model versions statistics
            if 'model_versions' in self.db.list_collection_names():
                model_versions_col = self.db.model_versions
                
                # Total versions
                stats['total_model_versions'] = model_versions_col.count_documents({})
                
                # Versions by model type
                pipeline = [
                    {'$group': {'_id': '$model_type', 'count': {'$sum': 1}}},
                    {'$sort': {'count': -1}}
                ]
                stats['versions_by_type'] = list(model_versions_col.aggregate(pipeline))
                
                # Active models
                stats['active_models'] = model_versions_col.count_documents({'is_active': True})
                
                # Recent versions (last 7 days)
                from datetime import datetime, timedelta
                week_ago = datetime.now() - timedelta(days=7)
                stats['recent_versions'] = model_versions_col.count_documents({
                    'created_at': {'$gte': week_ago}
                })
            else:
                stats['total_model_versions'] = 0
                stats['versions_by_type'] = []
                stats['active_models'] = 0
                stats['recent_versions'] = 0
            
            # Training events statistics
            if 'training_events' in self.db.list_collection_names():
                training_events_col = self.db.training_events
                
                # Total training events
                stats['total_training_events'] = training_events_col.count_documents({})
                
                # Successful vs failed training
                stats['successful_training'] = training_events_col.count_documents({'status': 'completed'})
                stats['failed_training'] = training_events_col.count_documents({'status': 'failed'})
                
                # Recent training events (last 24 hours)
                day_ago = datetime.now() - timedelta(days=1)
                stats['recent_training_events'] = training_events_col.count_documents({
                    'timestamp': {'$gte': day_ago}
                })
            else:
                stats['total_training_events'] = 0
                stats['successful_training'] = 0
                stats['failed_training'] = 0
                stats['recent_training_events'] = 0
            
            # Performance history statistics
            if 'performance_history' in self.db.list_collection_names():
                performance_col = self.db.performance_history
                stats['total_performance_records'] = performance_col.count_documents({})
            else:
                stats['total_performance_records'] = 0
            
            # Predictions statistics (existing collection)
            stats['total_predictions'] = self._predictions_col.count_documents({}) if self._predictions_col else 0
            
            return stats
            
        except Exception as e:
            print(f"Error getting adaptive learning stats: {e}")
            return {'error': str(e)}
    
    def save_model_version(self, version_data):
        """Save model version to database"""
        try:
            if self.db is None:
                raise Exception("Database not connected")
            
            collection = self.db.model_versions
            result = collection.insert_one(version_data)
            return result
        except Exception as e:
            print(f"Error saving model version: {e}")
            raise
    
    def get_model_versions(self, symbol=None, model_type=None, limit=50):
        """Get model versions with optional filtering"""
        try:
            if self.db is None:
                return []
            
            collection = self.db.model_versions
            query = {}
            
            if symbol:
                query['symbol'] = symbol
            if model_type:
                query['model_type'] = model_type
            
            cursor = collection.find(query).sort('created_at', -1).limit(limit)
            versions = []
            
            for doc in cursor:
                doc['_id'] = str(doc['_id'])
                versions.append(doc)
            
            return versions
        except Exception as e:
            print(f"Error getting model versions: {e}")
            return []
    
    def save_training_event(self, event_data):
        """Save training event to database"""
        try:
            if self.db is None:
                raise Exception("Database not connected")
            
            collection = self.db.training_events
            result = collection.insert_one(event_data)
            return result
        except Exception as e:
            print(f"Error saving training event: {e}")
            raise
    
    def get_training_events(self, symbol=None, model_type=None, limit=50):
        """Get training events with optional filtering"""
        try:
            if self.db is None:
                return []
            
            collection = self.db.training_events
            query = {}
            
            if symbol:
                query['symbol'] = symbol
            if model_type:
                query['model_type'] = model_type
            
            cursor = collection.find(query).sort('timestamp', -1).limit(limit)
            events = []
            
            for doc in cursor:
                doc['_id'] = str(doc['_id'])
                events.append(doc)
            
            return events
        except Exception as e:
            print(f"Error getting training events: {e}")
            return []
    
    def save_performance_record(self, performance_data):
        """Save performance record to database"""
        try:
            if self.db is None:
                raise Exception("Database not connected")
            
            collection = self.db.performance_history
            result = collection.insert_one(performance_data)
            return result
        except Exception as e:
            print(f"Error saving performance record: {e}")
            raise
    
    def get_performance_history(self, symbol=None, model_type=None, limit=100):
        """Get performance history with optional filtering"""
        try:
            if self.db is None:
                return []
            
            collection = self.db.performance_history
            query = {}
            
            if symbol:
                query['symbol'] = symbol
            if model_type:
                query['model_type'] = model_type
            
            cursor = collection.find(query).sort('timestamp', -1).limit(limit)
            records = []
            
            for doc in cursor:
                doc['_id'] = str(doc['_id'])
                records.append(doc)
            
            return records
        except Exception as e:
            print(f"Error getting performance history: {e}")
            return []
    
    def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()