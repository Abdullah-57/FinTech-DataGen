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
    
    def connect(self):
        """Connect to MongoDB"""
        try:
            mongodb_uri = os.getenv('MONGOURI')
            if not mongodb_uri:
                raise ValueError("MONGOURI environment variable not set")
            
            self.client = MongoClient(mongodb_uri)
            self.db = self.client.fintech
            
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
            if not self.db:
                raise Exception("Database not connected")
            
            collection = self.db.datasets
            result = collection.insert_one(dataset_data)
            return result
        except Exception as e:
            print(f"Error saving dataset: {e}")
            raise
    
    def get_dataset_by_id(self, dataset_id):
        """Get dataset by ID"""
        try:
            if not self.db:
                raise Exception("Database not connected")
            
            collection = self.db.datasets
            from bson import ObjectId
            return collection.find_one({"_id": ObjectId(dataset_id)})
        except Exception as e:
            print(f"Error getting dataset: {e}")
            return None
    
    def get_all_datasets(self):
        """Get all datasets"""
        try:
            if not self.db:
                raise Exception("Database not connected")
            
            collection = self.db.datasets
            datasets = list(collection.find().sort("generated_at", -1))
            
            # Convert ObjectId to string for JSON serialization
            for dataset in datasets:
                dataset['_id'] = str(dataset['_id'])
            
            return datasets
        except Exception as e:
            print(f"Error getting datasets: {e}")
            return []
    
    def get_recent_datasets(self, limit=10):
        """Get recent datasets"""
        try:
            if not self.db:
                raise Exception("Database not connected")
            
            collection = self.db.datasets
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
            if not self.db:
                raise Exception("Database not connected")
            
            collection = self.db.datasets
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
            if not self.db:
                raise Exception("Database not connected")
            
            collection = self.db.predictions
            result = collection.insert_one(prediction_data)
            return result
        except Exception as e:
            print(f"Error saving prediction: {e}")
            raise
    
    def get_recent_predictions(self, limit=10):
        """Get recent predictions"""
        try:
            if not self.db:
                raise Exception("Database not connected")
            
            collection = self.db.predictions
            predictions = list(collection.find().sort("created_at", -1).limit(limit))
            
            # Convert ObjectId to string
            for prediction in predictions:
                prediction['_id'] = str(prediction['_id'])
            
            return predictions
        except Exception as e:
            print(f"Error getting predictions: {e}")
            return []
    
    def count_datasets(self):
        """Count total datasets"""
        try:
            if not self.db:
                return 0
            
            collection = self.db.datasets
            return collection.count_documents({})
        except:
            return 0
    
    def count_records(self):
        """Count total records across all datasets"""
        try:
            if not self.db:
                return 0
            
            collection = self.db.datasets
            pipeline = [{"$group": {"_id": None, "total": {"$sum": "$records"}}}]
            result = list(collection.aggregate(pipeline))
            return result[0]['total'] if result else 0
        except:
            return 0
    
    def get_last_generated_date(self):
        """Get last generated date"""
        try:
            if not self.db:
                return None
            
            collection = self.db.datasets
            latest = collection.find_one(sort=[("generated_at", -1)])
            return latest['generated_at'].strftime('%Y-%m-%d %H:%M:%S') if latest else None
        except:
            return None
    
    def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()
