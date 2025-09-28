# FinTech DataGen Backend

This is the backend service for the FinTech DataGen application, which integrates the FinTech Data Curator for financial data generation and analysis.

## Features

- **Financial Data Generation**: Uses the integrated `fintech_data_curator.py` to generate comprehensive financial datasets
- **MongoDB Integration**: Stores generated datasets and predictions
- **REST API**: Provides endpoints for data generation, download, and analytics
- **Machine Learning**: Includes financial prediction capabilities
- **Error Handling**: Graceful handling of database connection issues

## Setup

### Prerequisites

- Python 3.8+
- MongoDB (optional - app works without it)

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables (optional):
```bash
# Create .env file with:
MONGOURI=mongodb://localhost:27017/fintech
PORT=5000
```

### Running the Application

1. Start the backend server:
```bash
python app.py
```

2. The API will be available at `http://localhost:5000`

## API Endpoints

- `GET /api/health` - Health check and system status
- `POST /api/generate` - Generate financial dataset
- `GET /api/datasets` - List all datasets
- `GET /api/datasets/<id>/csv` - Download dataset as CSV
- `GET /api/datasets/<id>/json` - Download dataset as JSON
- `GET /api/analytics` - Get analytics data
- `POST /api/predict` - Make financial predictions

## Testing

Run the integration test:
```bash
python test_integration.py
```

## File Structure

```
backend/
├── app.py                    # Main Flask application
├── fintech_data_curator.py   # Integrated data curator
├── database/
│   ├── mongodb.py           # MongoDB connection
│   └── __init__.py
├── ml_models/
│   ├── predictor.py         # ML prediction model
│   ├── feature_engineering.py
│   └── __init__.py
├── requirements.txt         # Python dependencies
├── test_integration.py      # Integration tests
└── README.md               # This file
```

## Error Handling

The application is designed to work even without MongoDB:
- If MongoDB is unavailable, the app will still generate data
- Database-dependent features will return appropriate error messages
- All core functionality (data generation) works independently

## Integration Notes

The `fintech_data_curator.py` has been fully integrated:
- Logging is configured to write to the backend directory
- All dependencies are included in requirements.txt
- Error handling is robust and user-friendly
- The curator can be used both standalone and through the API
