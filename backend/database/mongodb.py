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
                return None
            docs = []
            for p in prices:
                docs.append({
                    'symbol': symbol,
                    'exchange': exchange,
                    'date': p.get('date'),
                    'open': float(p.get('open_price') or p.get('open') or 0),
                    'high': float(p.get('high_price') or p.get('high') or 0),
                    'low': float(p.get('low_price') or p.get('low') or 0),
                    'close': float(p.get('close_price') or p.get('close') or 0),
                    'volume': int(p.get('volume') or 0)
                })
            result = collection.insert_many(docs, ordered=False)
            return result
        except Exception as e:
            print(f"Error saving historical prices: {e}")
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
            collection = self._metadata_col or self.db.metadata
            from pymongo import ReturnDocument
            updated = collection.find_one_and_update(
                {'symbol': symbol},
                {
                    '$set': {
                        'symbol': symbol,
                        'instrument_info': metadata.get('instrument_info'),
                        'data_sources': metadata.get('data_sources')
                    },
                    '$setOnInsert': {'created_at': datetime.now()},
                    '$push': {'update_logs': {'$each': metadata.get('update_logs', [])}}
                },
                upsert=True,
                return_document=ReturnDocument.AFTER
            )
            if updated and '_id' in updated:
                updated['_id'] = str(updated['_id'])
            return updated
        except Exception as e:
            print(f"Error upserting metadata: {e}")
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
    
    def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()
