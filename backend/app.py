from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from datetime import datetime
import os
import sys
import io
import csv
import json
import pandas as pd
from dotenv import load_dotenv
from database.mongodb import MongoDB
from ml_models.predictor import FinancialPredictor

# Import fintech_data_curator from the same directory
from fintech_data_curator import FinTechDataCurator

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize MongoDB connection
try:
    db = MongoDB()
    print("✅ MongoDB connection initialized")
except Exception as e:
    print(f"⚠️ MongoDB connection failed: {e}")
    db = None

# Initialize ML predictor
predictor = FinancialPredictor()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint to test backend connectivity"""
    try:
        # Test database connection
        if db is None:
            db_status = False
            stats = {
                'totalDatasets': 0,
                'totalRecords': 0,
                'lastGenerated': None
            }
        else:
            db_status = db.test_connection()
            stats = {
                'totalDatasets': db.count_datasets(),
                'totalRecords': db.count_records(),
                'lastGenerated': db.get_last_generated_date()
            }
        
        return jsonify({
            'status': 'healthy',
            'database': 'connected' if db_status else 'disconnected',
            'timestamp': datetime.now().isoformat(),
            'stats': stats
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/generate', methods=['POST'])
def generate_data():
    """Generate financial dataset using fintech_data_curator.py"""
    try:
        data = request.get_json()
        
        # Validate input
        required_fields = ['symbol', 'exchange', 'days']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Validate days is a positive integer
        try:
            days = int(data['days'])
            if days < 1:
                raise ValueError("Days must be positive")
        except (ValueError, TypeError):
            return jsonify({'error': 'Days must be a positive integer'}), 400
        
        # Initialize the FinTech Data Curator
        curator = FinTechDataCurator(days_history=days)
        
        # Generate dataset using the integrated curator
        dataset = curator.curate_dataset(data['symbol'], data['exchange'])
        
        # Convert MarketData objects to dictionaries for storage
        dataset_dict = []
        for item in dataset:
            # Helper function to handle NaN values
            def safe_value(value):
                if pd.isna(value):
                    return None
                return value
            
            dataset_dict.append({
                'symbol': item.symbol,
                'exchange': item.exchange,
                'date': item.date,
                'open_price': safe_value(item.open_price),
                'high_price': safe_value(item.high_price),
                'low_price': safe_value(item.low_price),
                'close_price': safe_value(item.close_price),
                'volume': int(item.volume) if not pd.isna(item.volume) else 0,
                'daily_return': safe_value(item.daily_return),
                'volatility': safe_value(item.volatility),
                'sma_5': safe_value(item.sma_5),
                'sma_20': safe_value(item.sma_20),
                'rsi': safe_value(item.rsi),
                'news_headlines': item.news_headlines if item.news_headlines else [],
                'news_sentiment_score': safe_value(item.news_sentiment_score)
            })
        
        # Save to database if available
        dataset_id = None
        if db is not None:
            try:
                result = db.save_dataset({
                    'symbol': data['symbol'],
                    'exchange': data['exchange'],
                    'days': days,
                    'records': len(dataset),
                    'generated_at': datetime.now(),
                    'data': dataset_dict
                })
                dataset_id = str(result.inserted_id)
            except Exception as e:
                print(f"Warning: Failed to save to database: {e}")
        
        return jsonify({
            'success': True,
            'message': 'Dataset generated successfully using FinTech Data Curator',
            'dataset_id': dataset_id,
            'records_count': len(dataset),
            'symbol': data['symbol'],
            'exchange': data['exchange'],
            'days': days
        }), 201
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/datasets/<dataset_id>/csv', methods=['GET'])
def download_csv(dataset_id):
    """Download dataset as CSV"""
    try:
        if db is None:
            return jsonify({'error': 'Database not available'}), 503
        
        dataset = db.get_dataset_by_id(dataset_id)
        if not dataset:
            return jsonify({'error': 'Dataset not found'}), 404
        
        # Create CSV content
        output = io.StringIO()
        fieldnames = [
            'symbol', 'exchange', 'date', 'open_price', 'high_price', 'low_price',
            'close_price', 'volume', 'daily_return', 'volatility', 'sma_5', 'sma_20',
            'rsi', 'news_headlines', 'news_sentiment_score'
        ]
        
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        for item in dataset['data']:
            # Properly format news headlines for CSV
            news_text = ' | '.join(item['news_headlines']) if item['news_headlines'] else ''
            # Replace problematic characters that could break CSV formatting
            news_text = news_text.replace('\n', ' ').replace('\r', ' ').replace('"', '""')
            
            writer.writerow({
                'symbol': item['symbol'],
                'exchange': item['exchange'],
                'date': item['date'],
                'open_price': round(item['open_price'], 4) if item['open_price'] is not None else '',
                'high_price': round(item['high_price'], 4) if item['high_price'] is not None else '',
                'low_price': round(item['low_price'], 4) if item['low_price'] is not None else '',
                'close_price': round(item['close_price'], 4) if item['close_price'] is not None else '',
                'volume': int(item['volume']) if item['volume'] is not None else 0,
                'daily_return': round(item['daily_return'], 6) if item['daily_return'] is not None else 0,
                'volatility': round(item['volatility'], 6) if item['volatility'] is not None else 0,
                'sma_5': round(item['sma_5'], 4) if item['sma_5'] is not None else '',
                'sma_20': round(item['sma_20'], 4) if item['sma_20'] is not None else '',
                'rsi': round(item['rsi'], 2) if item['rsi'] is not None else '',
                'news_headlines': news_text,
                'news_sentiment_score': round(item['news_sentiment_score'], 3) if item['news_sentiment_score'] is not None else 0
            })
        
        csv_content = output.getvalue()
        output.close()
        
        # Create file-like object
        csv_file = io.BytesIO(csv_content.encode('utf-8'))
        
        return send_file(
            csv_file,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f"{dataset['symbol']}_{dataset['exchange']}_data.csv"
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/datasets/<dataset_id>/json', methods=['GET'])
def download_json(dataset_id):
    """Download dataset as JSON"""
    try:
        if db is None:
            return jsonify({'error': 'Database not available'}), 503
        
        dataset = db.get_dataset_by_id(dataset_id)
        if not dataset:
            return jsonify({'error': 'Dataset not found'}), 404
        
        # Format data for JSON download
        json_data = []
        for item in dataset['data']:
            # Helper function to handle NaN values for JSON serialization
            def safe_float(value):
                if value is None or (isinstance(value, float) and str(value).lower() == 'nan'):
                    return None
                return round(float(value), 4) if isinstance(value, (int, float)) else value
            
            def safe_int(value):
                if value is None or (isinstance(value, float) and str(value).lower() == 'nan'):
                    return 0
                return int(value) if isinstance(value, (int, float)) else value
            
            json_data.append({
                'symbol': item['symbol'],
                'exchange': item['exchange'],
                'date': item['date'],
                'structured_data': {
                    'open_price': safe_float(item['open_price']),
                    'high_price': safe_float(item['high_price']),
                    'low_price': safe_float(item['low_price']),
                    'close_price': safe_float(item['close_price']),
                    'volume': safe_int(item['volume']),
                    'daily_return': round(item['daily_return'], 6) if item['daily_return'] is not None and str(item['daily_return']).lower() != 'nan' else 0,
                    'volatility': round(item['volatility'], 6) if item['volatility'] is not None and str(item['volatility']).lower() != 'nan' else 0,
                    'sma_5': safe_float(item['sma_5']),
                    'sma_20': safe_float(item['sma_20']),
                    'rsi': round(item['rsi'], 2) if item['rsi'] is not None and str(item['rsi']).lower() != 'nan' else 50
                },
                'unstructured_data': {
                    'news_headlines': item['news_headlines'] if item['news_headlines'] else [],
                    'news_sentiment_score': round(item['news_sentiment_score'], 3) if item['news_sentiment_score'] is not None and str(item['news_sentiment_score']).lower() != 'nan' else 0
                }
            })
        
        return jsonify(json_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """Get analytics data"""
    try:
        if db is None:
            datasets = []
            predictions = []
        else:
            datasets = db.get_recent_datasets(limit=10)
            predictions = db.get_recent_predictions(limit=10)
        
        # Calculate accuracy (placeholder)
        accuracy = predictor.calculate_accuracy()
        
        return jsonify({
            'datasets': datasets,
            'predictions': predictions,
            'accuracy': accuracy
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/predict', methods=['POST'])
def make_prediction():
    """Make financial prediction"""
    try:
        data = request.get_json()
        
        if 'symbol' not in data:
            return jsonify({'error': 'Missing symbol parameter'}), 400
        
        if db is None:
            return jsonify({'error': 'Database not available'}), 503
        
        # Get latest data for symbol
        latest_data = db.get_latest_data(data['symbol'])
        
        if not latest_data:
            return jsonify({'error': 'No data found for symbol'}), 404
        
        # Make prediction
        prediction = predictor.predict(latest_data)
        
        # Save prediction
        db.save_prediction({
            'symbol': data['symbol'],
            'prediction': prediction,
            'confidence': prediction.get('confidence', 0.0),
            'created_at': datetime.now()
        })
        
        return jsonify({
            'success': True,
            'prediction': prediction
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    """Get all datasets"""
    try:
        if db is None:
            return jsonify([]), 200
        datasets = db.get_all_datasets()
        return jsonify(datasets), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/datasets/<dataset_id>', methods=['GET'])
def get_dataset(dataset_id):
    """Get specific dataset by ID"""
    try:
        if db is None:
            return jsonify({'error': 'Database not available'}), 503
        dataset = db.get_dataset_by_id(dataset_id)
        if not dataset:
            return jsonify({'error': 'Dataset not found'}), 404
        return jsonify(dataset), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)