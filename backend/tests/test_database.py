#!/usr/bin/env python3
"""
Unit tests for MongoDB database operations in FinTech DataGen.

This module tests all critical database operations including:
- Connection management
- Dataset operations
- Prediction operations
- Historical price operations
- Metadata operations
- Error handling

Author: FinTech DataGen Team
Date: October 2025
"""

import unittest
import sys
import os
from unittest.mock import patch, MagicMock, Mock
from datetime import datetime

# Fix TensorFlow initialization issue if it appears
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd

# Add parent directory to path to import database module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.mongodb import MongoDB

class TestMongoDBConnection(unittest.TestCase):
    """Test MongoDB connection functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_uri = "mongodb://localhost:27017/test"
    
    @patch('database.mongodb.MongoClient')
    def test_successful_connection(self, mock_client):
        """Test successful MongoDB connection."""
        print("\n=== Testing Successful MongoDB Connection ===")
        
        # Mock successful connection
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.admin.command.return_value = {'ok': 1}
        
        with patch.dict(os.environ, {'MONGOURI': self.test_uri}):
            db = MongoDB()
            
            self.assertIsNotNone(db.client)
            self.assertIsNotNone(db.db)
            self.assertTrue(db.test_connection())
            
            print("MongoDB connection successful")
    
    @patch('database.mongodb.MongoClient')
    def test_connection_failure(self, mock_client):
        """Test MongoDB connection failure."""
        print("\n=== Testing MongoDB Connection Failure ===")
        
        # Mock connection failure
        mock_client.side_effect = Exception("Connection failed")
        
        with patch.dict(os.environ, {'MONGOURI': self.test_uri}):
            db = MongoDB()
            
            self.assertIsNone(db.client)
            self.assertIsNone(db.db)
            self.assertFalse(db.test_connection())
            
            print("MongoDB connection failure handled correctly")
    
    def test_missing_mongouri(self):
        """Test behavior when MONGOURI is not set."""
        print("\n=== Testing Missing MONGOURI ===")
        
        with patch.dict(os.environ, {}, clear=True):
            db = MongoDB()
            # MongoDB constructor now handles missing MONGOURI gracefully
            self.assertIsNone(db.client)
            self.assertIsNone(db.db)
            
            print("Missing MONGOURI handled correctly")

class TestMongoDBOperations(unittest.TestCase):
    """Test MongoDB database operations."""
    
    def setUp(self):
        """Set up mock MongoDB instance."""
        self.mock_db = MongoDB()
        
        # Mock database components
        self.mock_client = MagicMock()
        self.mock_db.client = self.mock_client
        self.mock_db.db = MagicMock()
        
        # Mock collections
        self.mock_datasets_col = MagicMock()
        self.mock_predictions_col = MagicMock()
        self.mock_historical_col = MagicMock()
        self.mock_metadata_col = MagicMock()
        
        self.mock_db._datasets_col = self.mock_datasets_col
        self.mock_db._predictions_col = self.mock_predictions_col
        self.mock_db._historical_col = self.mock_historical_col
        self.mock_db._metadata_col = self.mock_metadata_col
    
    def test_save_dataset(self):
        """Test saving dataset to database."""
        print("\n=== Testing Save Dataset ===")
        
        dataset_data = {
            'symbol': 'AAPL',
            'exchange': 'NASDAQ',
            'records': 30,
            'generated_at': datetime.now(),
            'data': []
        }
        
        mock_result = MagicMock()
        mock_result.inserted_id = 'test_id'
        self.mock_datasets_col.insert_one.return_value = mock_result
        
        result = self.mock_db.save_dataset(dataset_data)
        
        self.mock_datasets_col.insert_one.assert_called_once_with(dataset_data)
        self.assertEqual(result.inserted_id, 'test_id')
        
        print("Save dataset working correctly")
    
    def test_save_dataset_no_connection(self):
        """Test save dataset when no database connection."""
        print("\n=== Testing Save Dataset (No Connection) ===")
        
        self.mock_db.db = None
        
        with self.assertRaises(Exception):
            self.mock_db.save_dataset({})
        
        print("No connection handled correctly")
    
    def test_get_dataset_by_id(self):
        """Test getting dataset by ID."""
        print("\n=== Testing Get Dataset by ID ===")
        
        from bson import ObjectId
        
        mock_dataset = {
            '_id': ObjectId(),
            'symbol': 'AAPL',
            'exchange': 'NASDAQ',
            'records': 30
        }
        
        self.mock_datasets_col.find_one.return_value = mock_dataset
        
        # Test with valid ObjectId
        valid_id = str(mock_dataset['_id'])
        result = self.mock_db.get_dataset_by_id(valid_id)
        
        self.assertEqual(result['symbol'], mock_dataset['symbol'])
        self.mock_datasets_col.find_one.assert_called_once()
        
        print("Get dataset by ID working correctly")
    
    def test_get_all_datasets(self):
        """Test getting all datasets."""
        print("\n=== Testing Get All Datasets ===")
        
        mock_datasets = [
            {'_id': 'id1', 'symbol': 'AAPL'},
            {'_id': 'id2', 'symbol': 'MSFT'}
        ]
        
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = Mock(return_value=iter(mock_datasets))
        self.mock_datasets_col.find.return_value.sort.return_value = mock_cursor
        
        result = self.mock_db.get_all_datasets()
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['_id'], 'id1')
        
        print("Get all datasets working correctly")
    
    def test_save_historical_prices(self):
        """Test saving historical prices."""
        print("\n=== Testing Save Historical Prices ===")
        
        prices = [
            {
                'date': '2023-01-01',
                'open_price': 100.0,
                'high_price': 105.0,
                'low_price': 95.0,
                'close_price': 102.0,
                'volume': 1000000
            }
        ]
        
        mock_result = MagicMock()
        mock_result.inserted_ids = ['id1', 'id2']
        self.mock_historical_col.insert_many.return_value = mock_result
        
        result = self.mock_db.save_historical_prices('AAPL', 'NASDAQ', prices)
        
        self.mock_historical_col.insert_many.assert_called_once()
        self.assertEqual(result.inserted_ids, ['id1', 'id2'])
        
        print("Save historical prices working correctly")
    
    def test_get_prices(self):
        """Test getting historical prices."""
        print("\n=== Testing Get Prices ===")
        
        mock_prices = [
            {
                'symbol': 'AAPL',
                'date': '2023-01-01',
                'open': 100.0,
                'high': 105.0,
                'low': 95.0,
                'close': 102.0,
                'volume': 1000000
            }
        ]
        
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = Mock(return_value=iter(mock_prices))
        self.mock_historical_col.find.return_value.sort.return_value.limit.return_value = mock_cursor
        
        result = self.mock_db.get_prices(symbol='AAPL', limit=100)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['symbol'], 'AAPL')
        
        print("Get prices working correctly")
    
    def test_save_forecast(self):
        """Test saving forecast prediction."""
        print("\n=== Testing Save Forecast ===")
        
        forecast_data = {
            'symbol': 'AAPL',
            'model': 'moving_average',
            'forecast_horizon': 5,
            'predicted_values': [103.0, 104.0, 105.0, 106.0, 107.0],
            'metrics': {'rmse': 1.5, 'mae': 1.2, 'mape': 1.8},
            'created_at': datetime.now()
        }
        
        mock_result = MagicMock()
        mock_result.inserted_id = 'forecast_id'
        self.mock_predictions_col.insert_one.return_value = mock_result
        
        result = self.mock_db.save_forecast(forecast_data)
        
        self.mock_predictions_col.insert_one.assert_called_once_with(forecast_data)
        self.assertEqual(result.inserted_id, 'forecast_id')
        
        print("Save forecast working correctly")
    
    def test_get_predictions(self):
        """Test getting predictions with filters."""
        print("\n=== Testing Get Predictions ===")
        
        mock_predictions = [
            {
                '_id': 'pred1',
                'symbol': 'AAPL',
                'model': 'moving_average',
                'forecast_horizon': 5,
                'created_at': datetime.now()
            }
        ]
        
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = Mock(return_value=iter(mock_predictions))
        self.mock_predictions_col.find.return_value.sort.return_value.limit.return_value = mock_cursor
        
        result = self.mock_db.get_predictions(symbol='AAPL', model='moving_average', limit=10)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['symbol'], 'AAPL')
        
        print("Get predictions working correctly")
    
    def test_upsert_metadata(self):
        """Test upserting metadata."""
        print("\n=== Testing Upsert Metadata ===")
        
        metadata = {
            'instrument_info': {'name': 'Apple Inc.', 'sector': 'Technology'},
            'data_sources': ['Yahoo Finance', 'Google News'],
            'update_logs': [{'timestamp': datetime.now(), 'action': 'update'}]
        }
        
        mock_updated = {
            '_id': 'meta_id',
            'symbol': 'AAPL',
            'instrument_info': metadata['instrument_info'],
            'data_sources': metadata['data_sources'],
            'update_logs': metadata['update_logs']
        }
        
        self.mock_metadata_col.find_one_and_update.return_value = mock_updated
        
        result = self.mock_db.upsert_metadata('AAPL', metadata)
        
        self.mock_metadata_col.find_one_and_update.assert_called_once()
        self.assertEqual(result['symbol'], 'AAPL')
        
        print("Upsert metadata working correctly")
    
    def test_get_metadata(self):
        """Test getting metadata."""
        print("\n=== Testing Get Metadata ===")
        
        mock_metadata = {
            '_id': 'meta_id',
            'symbol': 'AAPL',
            'instrument_info': {'name': 'Apple Inc.'}
        }
        
        self.mock_metadata_col.find_one.return_value = mock_metadata
        
        result = self.mock_db.get_metadata('AAPL')
        
        self.assertEqual(result['symbol'], 'AAPL')
        
        print("Get metadata working correctly")
    
    def test_count_datasets(self):
        """Test counting datasets."""
        print("\n=== Testing Count Datasets ===")
        
        self.mock_datasets_col.count_documents.return_value = 25
        
        result = self.mock_db.count_datasets()
        
        self.assertEqual(result, 25)
        self.mock_datasets_col.count_documents.assert_called_once_with({})
        
        print("Count datasets working correctly")
    
    def test_count_records(self):
        """Test counting total records."""
        print("\n=== Testing Count Records ===")
        
        mock_aggregate_result = [{'_id': None, 'total': 1500}]
        self.mock_datasets_col.aggregate.return_value = mock_aggregate_result
        
        result = self.mock_db.count_records()
        
        self.assertEqual(result, 1500)
        
        print("Count records working correctly")
    
    def test_get_last_generated_date(self):
        """Test getting last generated date."""
        print("\n=== Testing Get Last Generated Date ===")
        
        mock_latest = {
            'generated_at': datetime(2023, 10, 1, 12, 0, 0)
        }
        
        self.mock_datasets_col.find_one.return_value = mock_latest
        
        result = self.mock_db.get_last_generated_date()
        
        self.assertEqual(result, '2023-10-01 12:00:00')
        
        print("Get last generated date working correctly")
    
    def test_get_last_generated_date_none(self):
        """Test getting last generated date when no data exists."""
        print("\n=== Testing Get Last Generated Date (None) ===")
        
        self.mock_datasets_col.find_one.return_value = None
        
        result = self.mock_db.get_last_generated_date()
        
        self.assertIsNone(result)
        
        print("No data handled correctly")
    
    def test_close_connection(self):
        """Test closing database connection."""
        print("\n=== Testing Close Connection ===")
        
        self.mock_db.close()
        
        self.mock_client.close.assert_called_once()
        
        print("Close connection working correctly")

class TestMongoDBErrorHandling(unittest.TestCase):
    """Test MongoDB error handling."""
    
    def setUp(self):
        """Set up mock MongoDB instance."""
        self.mock_db = MongoDB()
        self.mock_db.db = MagicMock()
        self.mock_db._datasets_col = MagicMock()
        self.mock_db._predictions_col = MagicMock()
        self.mock_db._historical_col = MagicMock()
    
    def test_database_operation_exception(self):
        """Test handling of database operation exceptions."""
        print("\n=== Testing Database Operation Exception ===")
        
        # Mock database operation to raise exception
        self.mock_db._datasets_col.insert_one.side_effect = Exception("Database error")
        
        with self.assertRaises(Exception):
            self.mock_db.save_dataset({})
        
        print("Database exception handled correctly")
    
    def test_get_predictions_exception(self):
        """Test handling of get predictions exception."""
        print("\n=== Testing Get Predictions Exception ===")
        
        # Mock database operation to raise exception
        self.mock_db._predictions_col.find.side_effect = Exception("Query error")
        
        result = self.mock_db.get_predictions()
        
        self.assertEqual(result, [])
        
        print("Get predictions exception handled correctly")
    
    def test_get_prices_exception(self):
        """Test handling of get prices exception."""
        print("\n=== Testing Get Prices Exception ===")
        
        # Mock database operation to raise exception
        self.mock_db._historical_col.find.side_effect = Exception("Query error")
        
        result = self.mock_db.get_prices(symbol='AAPL')
        
        self.assertEqual(result, [])
        
        print("Get prices exception handled correctly")

if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
