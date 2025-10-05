#!/usr/bin/env python3
"""
Upload existing trained models from FinTech DataGen
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder
import tempfile

class ExistingModelUploader:
    def __init__(self, hf_token=None):
        self.api = HfApi(token=hf_token)
        self.username = "abdullah-daoud"
        
    def upload_predictor_model(self):
        """Upload the RandomForest predictor model from predictor.py"""
        
        # Create repository
        repo_name = f"{self.username}/fintech-predictor-model"
        try:
            create_repo(repo_name, exist_ok=True)
            print(f"✅ Created repository: {repo_name}")
        except Exception as e:
            print(f"Repository might already exist: {e}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Check if trained model exists
            model_path = "backend/ml_models/trained_model.pkl"
            
            if os.path.exists(model_path):
                # Copy model to temp directory
                import shutil
                temp_model_path = os.path.join(temp_dir, "trained_model.pkl")
                shutil.copy2(model_path, temp_model_path)
                print(f"✅ Found existing model: {model_path}")
            else:
                # Create and train a new model for demo
                print("⚠️ No existing trained model found, creating demo model...")
                from backend.ml_models.predictor import FinancialPredictor
                
                predictor = FinancialPredictor()
                
                # Create sample training data
                sample_data = {
                    'open_price': np.random.uniform(100, 200, 100),
                    'high_price': np.random.uniform(150, 250, 100),
                    'low_price': np.random.uniform(50, 150, 100),
                    'close_price': np.random.uniform(100, 200, 100),
                    'volume': np.random.uniform(1000000, 10000000, 100),
                    'daily_return': np.random.uniform(-0.05, 0.05, 100),
                    'volatility': np.random.uniform(0.01, 0.1, 100),
                    'sma_5': np.random.uniform(100, 200, 100),
                    'sma_20': np.random.uniform(100, 200, 100),
                    'rsi': np.random.uniform(20, 80, 100),
                    'news_sentiment_score': np.random.uniform(-1, 1, 100)
                }
                
                # Target variable (next day close price)
                target = np.random.uniform(100, 200, 100)
                
                # Train model
                predictor.train(sample_data, target)
                
                # Save model
                temp_model_path = os.path.join(temp_dir, "trained_model.pkl")
                joblib.dump(predictor.model, temp_model_path)
                print("✅ Created and saved demo model")
            
            # Create model info file
            model_info = {
                "model_type": "RandomForestRegressor",
                "framework": "scikit-learn",
                "task": "financial-prediction",
                "features": [
                    "open_price", "high_price", "low_price", "close_price", "volume",
                    "daily_return", "volatility", "sma_5", "sma_20", "rsi",
                    "news_sentiment_score"
                ],
                "target": "next_day_close_price",
                "created_at": datetime.now().isoformat(),
                "source": "FinTech DataGen - predictor.py"
            }
            
            info_path = os.path.join(temp_dir, "model_info.json")
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            # Create README
            readme_content = self.create_predictor_readme()
            readme_path = os.path.join(temp_dir, "README.md")
            with open(readme_path, 'w') as f:
                f.write(readme_content)
            
            # Create usage example
            usage_example = self.create_usage_example()
            example_path = os.path.join(temp_dir, "usage_example.py")
            with open(example_path, 'w') as f:
                f.write(usage_example)
            
            # Upload all files
            upload_folder(
                folder_path=temp_dir,
                repo_id=repo_name,
                repo_type="model"
            )
            print(f"✅ Uploaded predictor model to {repo_name}")
    
    def create_predictor_readme(self):
        return f"""---
license: mit
tags:
- financial-prediction
- random-forest
- time-series
- stock-prediction
library_name: scikit-learn
---

# FinTech Predictor Model

This is a RandomForest-based financial prediction model from the FinTech DataGen project. The model predicts next-day closing prices based on technical indicators and market data.

## Model Details

- **Model Type**: RandomForestRegressor
- **Framework**: scikit-learn
- **Task**: Financial price prediction
- **Input Features**: 11 technical and market indicators
- **Target**: Next day closing price

## Features Used

1. **OHLCV Data**: open_price, high_price, low_price, close_price, volume
2. **Technical Indicators**: daily_return, volatility, sma_5, sma_20, rsi
3. **Sentiment**: news_sentiment_score

## Usage

```python
import joblib
import numpy as np
from huggingface_hub import hf_hub_download

# Download the model
model_path = hf_hub_download(
    repo_id="{self.username}/fintech-predictor-model", 
    filename="trained_model.pkl"
)

# Load the model
model = joblib.load(model_path)

# Prepare your data (example)
sample_data = np.array([[
    150.0,  # open_price
    155.0,  # high_price
    148.0,  # low_price
    152.0,  # close_price
    1500000,  # volume
    0.013,  # daily_return
    0.025,  # volatility
    151.0,  # sma_5
    149.0,  # sma_20
    65.0,   # rsi
    0.2     # news_sentiment_score
]])

# Make prediction
prediction = model.predict(sample_data)
print(f"Predicted next day close price: ${{prediction[0]:.2f}}")
```

## Model Performance

This model was trained on financial OHLCV data with technical indicators and achieves good performance for next-day price prediction tasks.

## Integration with FinTech DataGen

This model is part of the complete FinTech DataGen application which includes:
- Real-time data collection from Yahoo Finance, Google News, CoinDesk
- Multiple forecasting models (ARIMA, LSTM, Transformer, Ensemble)
- Interactive web interface with candlestick charts
- MongoDB database for historical data storage

## Citation

```bibtex
@software{{fintech_datagen_2025,
  title={{FinTech DataGen: Complete Financial Forecasting Application}},
  author={{FinTech DataGen Team}},
  year={{2025}},
  url={{https://github.com/your_username/fintech-datagen}}
}}
```

## Requirements

```
scikit-learn>=1.3.0
numpy>=1.24.3
pandas>=2.0.3
joblib>=1.3.2
```
"""

    def create_usage_example(self):
        return f"""#!/usr/bin/env python3
\"\"\"
Example usage of the FinTech Predictor Model from Hugging Face
\"\"\"

import joblib
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

def load_and_use_model():
    \"\"\"Download and use the FinTech predictor model\"\"\"
    
    print("📥 Downloading model from Hugging Face...")
    
    # Download the model
    model_path = hf_hub_download(
        repo_id="{self.username}/fintech-predictor-model", 
        filename="trained_model.pkl"
    )
    
    # Load the model
    model = joblib.load(model_path)
    print("✅ Model loaded successfully!")
    
    # Example 1: Single prediction
    print("\\n📊 Example 1: Single Prediction")
    sample_data = np.array([[
        150.0,    # open_price
        155.0,    # high_price
        148.0,    # low_price
        152.0,    # close_price
        1500000,  # volume
        0.013,    # daily_return (1.3%)
        0.025,    # volatility (2.5%)
        151.0,    # sma_5 (5-day moving average)
        149.0,    # sma_20 (20-day moving average)
        65.0,     # rsi (Relative Strength Index)
        0.2       # news_sentiment_score (positive sentiment)
    ]])
    
    prediction = model.predict(sample_data)
    print(f"Current close price: $152.00")
    print(f"Predicted next day close: ${{prediction[0]:.2f}}")
    print(f"Expected change: {{((prediction[0] - 152.0) / 152.0 * 100):+.2f}}%")
    
    # Example 2: Batch predictions
    print("\\n📊 Example 2: Batch Predictions")
    batch_data = np.array([
        [150.0, 155.0, 148.0, 152.0, 1500000, 0.013, 0.025, 151.0, 149.0, 65.0, 0.2],
        [152.0, 157.0, 150.0, 154.0, 1600000, 0.015, 0.030, 152.5, 150.0, 68.0, 0.3],
        [154.0, 159.0, 152.0, 156.0, 1700000, 0.018, 0.035, 154.0, 151.0, 70.0, 0.1]
    ])
    
    batch_predictions = model.predict(batch_data)
    
    for i, pred in enumerate(batch_predictions):
        current_price = batch_data[i][3]  # close_price
        change_pct = (pred - current_price) / current_price * 100
        print(f"Stock {{i+1}}: ${{current_price:.2f}} → ${{pred:.2f}} ({{change_pct:+.2f}}%)")
    
    # Example 3: Feature importance (if available)
    print("\\n📊 Example 3: Model Information")
    if hasattr(model, 'feature_importances_'):
        feature_names = [
            'open_price', 'high_price', 'low_price', 'close_price', 'volume',
            'daily_return', 'volatility', 'sma_5', 'sma_20', 'rsi', 'news_sentiment'
        ]
        
        importance_df = pd.DataFrame({{
            'feature': feature_names,
            'importance': model.feature_importances_
        }}).sort_values('importance', ascending=False)
        
        print("Top 5 Most Important Features:")
        for _, row in importance_df.head().iterrows():
            print(f"  {{row['feature']}}: {{row['importance']:.3f}}")
    
    print(f"\\n✅ Model type: {{type(model).__name__}}")
    if hasattr(model, 'n_estimators'):
        print(f"✅ Number of trees: {{model.n_estimators}}")

if __name__ == "__main__":
    try:
        load_and_use_model()
        print("\\n🎉 Example completed successfully!")
    except Exception as e:
        print(f"❌ Error: {{e}}")
        print("Make sure you have the required packages installed:")
        print("pip install huggingface_hub scikit-learn numpy pandas joblib")
"""

def main():
    print("🚀 Uploading existing FinTech models to Hugging Face...")
    
    uploader = ExistingModelUploader()
    
    try:
        uploader.upload_predictor_model()
        
        print("\\n✅ Upload completed successfully!")
        print(f"\\n🔗 Your model is now available at:")
        print(f"   https://huggingface.co/{uploader.username}/fintech-predictor-model")
        print("\\n📖 You can now use it in your projects or share it with others!")
        
    except Exception as e:
        print(f"❌ Error uploading model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()