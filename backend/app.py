from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import os
from dotenv import load_dotenv
from database.mongodb import MongoDB
from ml_models.predictor import FinancialPredictor

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize MongoDB connection
db = MongoDB()

# Initialize ML predictor
predictor = FinancialPredictor()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint to test backend connectivity"""
    try:
        # Test database connection
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
    """Generate financial dataset"""
    try:
        data = request.get_json()
        
        # Validate input
        required_fields = ['symbol', 'exchange', 'days', 'dataType']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Generate dataset using existing curator
        from fintech_data_curator import FinTechDataCurator
        
        curator = FinTechDataCurator(days_history=data['days'])
        dataset = curator.curate_dataset(data['symbol'], data['exchange'])
        
        # Save to database
        result = db.save_dataset({
            'symbol': data['symbol'],
            'exchange': data['exchange'],
            'dataType': data['dataType'],
            'days': data['days'],
            'records': len(dataset),
            'generated_at': datetime.now(),
            'data': [item.__dict__ for item in dataset]
        })
        
        return jsonify({
            'success': True,
            'message': 'Dataset generated successfully',
            'dataset_id': str(result.inserted_id),
            'records_count': len(dataset),
            'symbol': data['symbol'],
            'exchange': data['exchange']
        }), 201
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """Get analytics data"""
    try:
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
        datasets = db.get_all_datasets()
        return jsonify(datasets), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/datasets/<dataset_id>', methods=['GET'])
def get_dataset(dataset_id):
    """Get specific dataset by ID"""
    try:
        dataset = db.get_dataset_by_id(dataset_id)
        if not dataset:
            return jsonify({'error': 'Dataset not found'}), 404
        return jsonify(dataset), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
