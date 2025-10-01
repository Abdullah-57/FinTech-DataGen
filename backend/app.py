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
from ml_models.forecasting import (
    moving_average_forecast,
    arima_forecast,
    lstm_forecast,
    transformer_forecast,
    MovingAverageForecaster,
    ARIMAForecaster,
    LSTMForecaster,
    TransformerForecaster,
    EnsembleAverageForecaster,
    train_test_split_series
)

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
                # Also persist curated OHLCV into historical_prices collection
                try:
                    db.save_historical_prices(
                        symbol=data['symbol'],
                        exchange=data['exchange'],
                        prices=dataset_dict
                    )
                except Exception as e:
                    print(f"Warning: Failed to save historical prices: {e}")
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

# ---------------------- New API Endpoints ----------------------
@app.route('/api/prices', methods=['GET'])
def get_prices():
    """Query historical OHLCV for visualization.
    Query params: symbol (required), start_date, end_date, limit
    """
    try:
        if db is None:
            return jsonify({'error': 'Database not available'}), 503
        symbol = request.args.get('symbol')
        if not symbol:
            return jsonify({'error': 'symbol is required'}), 400
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        limit = request.args.get('limit', default=500)
        rows = db.get_prices(symbol=symbol, start_date=start_date, end_date=end_date, limit=limit)
        return jsonify({
            'symbol': symbol,
            'rows': rows
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions', methods=['POST'])
def post_prediction():
    """Insert new forecast after model runs.
    Body: { symbol, model, forecast_horizon, predicted_values: [...], metadata? }
    """
    try:
        if db is None:
            return jsonify({'error': 'Database not available'}), 503
        payload = request.get_json(force=True)
        required = ['symbol', 'model', 'forecast_horizon', 'predicted_values']
        for r in required:
            if r not in payload:
                return jsonify({'error': f'Missing field: {r}'}), 400
        doc = {
            'symbol': payload['symbol'],
            'model': payload['model'],
            'forecast_horizon': payload['forecast_horizon'],
            'predicted_values': payload['predicted_values'],
            'created_at': datetime.now()
        }
        # include optional fields
        for opt in ['metrics', 'notes']:
            if opt in payload:
                doc[opt] = payload[opt]
        result = db.save_forecast(doc)
        return jsonify({'id': str(result.inserted_id)}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions', methods=['GET'])
def list_predictions():
    """Query predictions for visualization. Query: symbol, horizon, limit"""
    try:
        if db is None:
            return jsonify([]), 200
        symbol = request.args.get('symbol')
        horizon = request.args.get('horizon')
        limit = request.args.get('limit', default=50)
        docs = db.get_predictions(symbol=symbol, horizon=horizon, limit=limit)
        return jsonify(docs), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/metadata', methods=['POST'])
def upsert_metadata():
    """Upsert instrument metadata including data sources and update logs."""
    try:
        if db is None:
            return jsonify({'error': 'Database not available'}), 503
        payload = request.get_json(force=True)
        symbol = payload.get('symbol')
        if not symbol:
            return jsonify({'error': 'symbol is required'}), 400
        updated = db.upsert_metadata(symbol, {
            'instrument_info': payload.get('instrument_info'),
            'data_sources': payload.get('data_sources'),
            'update_logs': payload.get('update_logs', [])
        })
        return jsonify(updated or {}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/metadata', methods=['GET'])
def get_metadata():
    """Get metadata for one or all instruments. Query: symbol (optional)"""
    try:
        if db is None:
            return jsonify([]), 200
        symbol = request.args.get('symbol')
        doc_or_docs = db.get_metadata(symbol)
        return jsonify(doc_or_docs if doc_or_docs is not None else ({} if symbol else [])), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ---------------------- Simple public endpoints ----------------------
@app.route('/get_historical', methods=['GET'])
def get_historical_public():
    """Alias public endpoint for historical prices: /get_historical?symbol=XYZ&limit=300"""
    try:
        if db is None:
            return jsonify({'error': 'Database not available'}), 503
        symbol = request.args.get('symbol')
        limit = request.args.get('limit', default=300)
        if not symbol:
            return jsonify({'error': 'symbol is required'}), 400
        rows = db.get_prices(symbol=symbol, limit=limit)
        return jsonify({'symbol': symbol, 'rows': rows}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def _parse_horizon_to_hours(h: str) -> int:
    try:
        s = (h or '').strip().lower()
        if s.endswith('h'):
            return max(1, int(s[:-1]))
        if s.endswith('d'):
            return max(1, int(s[:-1])) * 24
        # default assume hours
        return max(1, int(s))
    except:
        return 24

@app.route('/get_forecast', methods=['GET'])
def get_forecast_public():
    """Public endpoint to get forecast preview: /get_forecast?symbol=XYZ&horizon=24h&models=ma,arima,lstm&ensemble=true"""
    try:
        if db is None:
            return jsonify({'error': 'Database not available'}), 503
        symbol = request.args.get('symbol')
        if not symbol:
            return jsonify({'error': 'symbol is required'}), 400
        horizon_raw = request.args.get('horizon', default='24h')
        preview_hours = _parse_horizon_to_hours(horizon_raw)
        models_param = request.args.get('models')  # e.g., "ma,arima,lstm,transformer"
        models = [m.strip() for m in models_param.split(',')] if models_param else ['ma', 'arima', 'lstm']
        ensemble = str(request.args.get('ensemble', 'false')).lower() in ['1', 'true', 'yes']

        # Map hours->days (daily data)
        preview_days = max(1, int(round(preview_hours / 24)))

        # fetch historical series
        rows = db.get_prices(symbol=symbol, limit=2000)
        if not rows or len(rows) < 20:
            return jsonify({'error': 'Insufficient historical data'}), 400
        df = pd.DataFrame(rows).sort_values('date')
        series = pd.Series(df['close'].values, index=pd.to_datetime(df['date']))
        train_series, test_series = train_test_split_series(series)

        results = []
        previews = []

        if 'ma' in models:
            ma_model = MovingAverageForecaster(window=5)
            ma_model.fit(train_series)
            eval_res = ma_model.evaluate(test_series)
            results.append({'model': 'moving_average', **eval_res})
            previews.append({'model': 'moving_average', 'horizon_hours': preview_hours, 'horizon_days': preview_days, 'predicted_values': ma_model.predict(preview_days)})

        if 'arima' in models:
            arima_model = ARIMAForecaster(order=(1,1,1))
            arima_model.fit(train_series)
            eval_res = arima_model.evaluate(test_series)
            results.append({'model': 'ARIMA(1, 1, 1)', **eval_res})
            previews.append({'model': 'ARIMA(1, 1, 1)', 'horizon_hours': preview_hours, 'horizon_days': preview_days, 'predicted_values': arima_model.predict(preview_days)})

        if 'lstm' in models:
            lstm_model = LSTMForecaster(lookback=10, epochs=20)
            lstm_model.fit(train_series)
            eval_res = lstm_model.evaluate(test_series)
            results.append({'model': 'LSTM', **eval_res})
            previews.append({'model': 'LSTM', 'horizon_hours': preview_hours, 'horizon_days': preview_days, 'predicted_values': lstm_model.predict(preview_days)})

        if 'transformer' in models:
            trans_model = TransformerForecaster(lookback=24, epochs=20)
            trans_model.fit(train_series)
            eval_res = trans_model.evaluate(test_series)
            results.append({'model': 'Transformer', **eval_res})
            previews.append({'model': 'Transformer', 'horizon_hours': preview_hours, 'horizon_days': preview_days, 'predicted_values': trans_model.predict(preview_days)})

        if ensemble and results:
            selected = []
            if 'ma' in models:
                selected.append(MovingAverageForecaster(window=5))
            if 'arima' in models:
                selected.append(ARIMAForecaster(order=(1,1,1)))
            if 'lstm' in models:
                selected.append(LSTMForecaster(lookback=10, epochs=20))
            if 'transformer' in models:
                selected.append(TransformerForecaster(lookback=24, epochs=20))
            ens = EnsembleAverageForecaster(selected)
            ens.fit(train_series)
            eval_res = ens.evaluate(test_series)
            results.append({'model': 'EnsembleAverage', **eval_res})
            previews.append({'model': 'EnsembleAverage', 'horizon_hours': preview_hours, 'horizon_days': preview_days, 'predicted_values': ens.predict(preview_days)})

        last_date = pd.to_datetime(series.index[-1])
        preview_dates = [(last_date + pd.Timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(preview_days)]

        return jsonify({
            'symbol': symbol,
            'results': results,
            'preview': {
                'dates': preview_dates,
                'models': previews,
                'horizon': horizon_raw
            }
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/forecast/run', methods=['POST'])
def run_forecast():
    """Run selected forecasting models on a symbol's close prices and store outputs.
    Body: { symbol: str, models: ["ma", "arima", "lstm"], ma_window?, arima_order? [p,d,q] }
    """
    try:
        if db is None:
            return jsonify({'error': 'Database not available'}), 503
        payload = request.get_json(force=True)
        symbol = payload.get('symbol')
        models = payload.get('models', ['ma', 'arima', 'lstm'])
        preview_hours = int(payload.get('preview_horizon_hours', 24))
        # Map hours to whole days since data is daily
        preview_days = max(1, int(round(preview_hours / 24)))
        if not symbol:
            return jsonify({'error': 'symbol is required'}), 400

        # fetch historical close series
        rows = db.get_prices(symbol=symbol, limit=2000)
        if not rows or len(rows) < 20:
            return jsonify({'error': 'Insufficient historical data'}), 400
        df = pd.DataFrame(rows)
        df = df.sort_values('date')
        series = pd.Series(df['close'].values, index=pd.to_datetime(df['date']))
        train_series, test_series = train_test_split_series(series)

        results = []
        preview_outputs = []
        # moving average
        if 'ma' in models:
            window = int(payload.get('ma_window', 5))
            ma_model = MovingAverageForecaster(window=window)
            ma_model.fit(train_series)
            eval_res = ma_model.evaluate(test_series)
            ma_res = {
                'model': 'moving_average',
                **eval_res
            }
            db.save_forecast({
                'symbol': symbol,
                'model': 'moving_average',
                'forecast_horizon': ma_res['forecast_horizon'],
                'predicted_values': ma_res['predicted_values'],
                'metrics': ma_res['metrics'],
                'y_true': ma_res['y_true'],
                'created_at': datetime.now()
            })
            results.append(ma_res)
            # Preview future horizon from fitted model
            ma_preview = {
                'model': 'moving_average',
                'horizon_hours': preview_hours,
                'horizon_days': preview_days,
                'predicted_values': ma_model.predict(preview_days)
            }
            preview_outputs.append(ma_preview)

        # ARIMA
        if 'arima' in models:
            order = payload.get('arima_order', [1, 1, 1])
            if not isinstance(order, (list, tuple)) or len(order) != 3:
                order = [1, 1, 1]
            arima_model = ARIMAForecaster(order=tuple(int(x) for x in order))
            arima_model.fit(train_series)
            eval_res = arima_model.evaluate(test_series)
            arima_res = {
                'model': f'ARIMA{tuple(int(x) for x in order)}',
                **eval_res
            }
            db.save_forecast({
                'symbol': symbol,
                'model': arima_res['model'],
                'forecast_horizon': arima_res['forecast_horizon'],
                'predicted_values': arima_res['predicted_values'],
                'metrics': arima_res['metrics'],
                'y_true': arima_res['y_true'],
                'created_at': datetime.now()
            })
            results.append(arima_res)
            arima_preview = {
                'model': arima_res['model'],
                'horizon_hours': preview_hours,
                'horizon_days': preview_days,
                'predicted_values': arima_model.predict(preview_days)
            }
            preview_outputs.append(arima_preview)

        # LSTM
        if 'lstm' in models:
            lookback = int(payload.get('lstm_lookback', 10))
            lstm_model = LSTMForecaster(lookback=lookback, epochs=int(payload.get('lstm_epochs', 40)))
            lstm_model.fit(train_series)
            eval_res = lstm_model.evaluate(test_series)
            lstm_res = {
                'model': 'LSTM',
                **eval_res
            }
            db.save_forecast({
                'symbol': symbol,
                'model': lstm_res['model'],
                'forecast_horizon': lstm_res['forecast_horizon'],
                'predicted_values': lstm_res['predicted_values'],
                'metrics': lstm_res['metrics'],
                'y_true': lstm_res['y_true'],
                'created_at': datetime.now()
            })
            results.append(lstm_res)
            lstm_preview = {
                'model': 'LSTM',
                'horizon_hours': preview_hours,
                'horizon_days': preview_days,
                'predicted_values': lstm_model.predict(preview_days)
            }
            preview_outputs.append(lstm_preview)

        # Transformer (optional)
        if 'transformer' in models:
            t_lookback = int(payload.get('transformer_lookback', 24))
            t_epochs = int(payload.get('transformer_epochs', 30))
            t_heads = int(payload.get('transformer_heads', 2))
            t_d_model = int(payload.get('transformer_d_model', 32))
            t_ff = int(payload.get('transformer_ff_dim', 64))
            t_dropout = float(payload.get('transformer_dropout', 0.1))
            trans_model = TransformerForecaster(
                lookback=t_lookback,
                d_model=t_d_model,
                num_heads=t_heads,
                ff_dim=t_ff,
                epochs=t_epochs,
                dropout=t_dropout
            )
            trans_model.fit(train_series)
            eval_res = trans_model.evaluate(test_series)
            trans_res = {
                'model': 'Transformer',
                **eval_res
            }
            db.save_forecast({
                'symbol': symbol,
                'model': trans_res['model'],
                'forecast_horizon': trans_res['forecast_horizon'],
                'predicted_values': trans_res['predicted_values'],
                'metrics': trans_res['metrics'],
                'y_true': trans_res['y_true'],
                'created_at': datetime.now()
            })
            results.append(trans_res)
            trans_preview = {
                'model': 'Transformer',
                'horizon_hours': preview_hours,
                'horizon_days': preview_days,
                'predicted_values': trans_model.predict(preview_days)
            }
            preview_outputs.append(trans_preview)

        # Optional ensemble: average of selected models
        if payload.get('ensemble'):
            selected = []
            if 'ma' in models:
                selected.append(MovingAverageForecaster(window=int(payload.get('ma_window', 5))))
            if 'arima' in models:
                order = payload.get('arima_order', [1,1,1])
                if not isinstance(order, (list, tuple)) or len(order) != 3:
                    order = [1,1,1]
                selected.append(ARIMAForecaster(order=tuple(int(x) for x in order)))
            if 'lstm' in models:
                selected.append(LSTMForecaster(lookback=int(payload.get('lstm_lookback', 10)), epochs=int(payload.get('lstm_epochs', 40))))
            if 'transformer' in models:
                selected.append(TransformerForecaster(
                    lookback=int(payload.get('transformer_lookback', 24)),
                    d_model=int(payload.get('transformer_d_model', 32)),
                    num_heads=int(payload.get('transformer_heads', 2)),
                    ff_dim=int(payload.get('transformer_ff_dim', 64)),
                    epochs=int(payload.get('transformer_epochs', 30)),
                    dropout=float(payload.get('transformer_dropout', 0.1))
                ))
            if selected:
                ens = EnsembleAverageForecaster(selected)
                ens.fit(train_series)
                eval_res = ens.evaluate(test_series)
                ens_res = {
                    'model': 'EnsembleAverage',
                    **eval_res
                }
                db.save_forecast({
                    'symbol': symbol,
                    'model': 'EnsembleAverage',
                    'forecast_horizon': ens_res['forecast_horizon'],
                    'predicted_values': ens_res['predicted_values'],
                    'metrics': ens_res['metrics'],
                    'y_true': ens_res['y_true'],
                    'created_at': datetime.now()
                })
                results.append(ens_res)

        # build preview dates for plotting
        last_date = pd.to_datetime(series.index[-1])
        preview_dates = [(last_date + pd.Timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(preview_days)]
        return jsonify({
            'symbol': symbol,
            'results': results,
            'preview': {
                'dates': preview_dates,
                'models': preview_outputs
            }
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)