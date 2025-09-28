import pandas as pd
import numpy as np
from typing import List, Dict, Any

class FeatureEngineer:
    """Feature engineering utilities for financial data"""
    
    def __init__(self):
        self.feature_names = []
    
    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators from OHLCV data"""
        try:
            # Moving averages
            df['sma_5'] = df['close_price'].rolling(window=5).mean()
            df['sma_20'] = df['close_price'].rolling(window=20).mean()
            df['ema_12'] = df['close_price'].ewm(span=12).mean()
            df['ema_26'] = df['close_price'].ewm(span=26).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # RSI
            df['rsi'] = self._calculate_rsi(df['close_price'])
            
            # Bollinger Bands
            df['bb_middle'] = df['close_price'].rolling(window=20).mean()
            bb_std = df['close_price'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_width'] = df['bb_upper'] - df['bb_lower']
            df['bb_position'] = (df['close_price'] - df['bb_lower']) / df['bb_width']
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Price change indicators
            df['daily_return'] = df['close_price'].pct_change()
            df['volatility'] = df['daily_return'].rolling(window=20).std()
            df['price_change_5d'] = df['close_price'].pct_change(5)
            df['price_change_20d'] = df['close_price'].pct_change(20)
            
            return df
            
        except Exception as e:
            print(f"Error creating technical indicators: {e}")
            return df
    
    def create_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create sentiment-based features"""
        try:
            # Sentiment momentum
            df['sentiment_ma_5'] = df['news_sentiment_score'].rolling(window=5).mean()
            df['sentiment_ma_20'] = df['news_sentiment_score'].rolling(window=20).mean()
            
            # Sentiment volatility
            df['sentiment_volatility'] = df['news_sentiment_score'].rolling(window=10).std()
            
            # Sentiment change
            df['sentiment_change'] = df['news_sentiment_score'].diff()
            
            # Sentiment extremes
            df['sentiment_extreme_positive'] = (df['news_sentiment_score'] > 0.8).astype(int)
            df['sentiment_extreme_negative'] = (df['news_sentiment_score'] < -0.8).astype(int)
            
            return df
            
        except Exception as e:
            print(f"Error creating sentiment features: {e}")
            return df
    
    def create_lag_features(self, df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
        """Create lagged features"""
        try:
            for col in columns:
                if col in df.columns:
                    for lag in lags:
                        df[f'{col}_lag_{lag}'] = df[col].shift(lag)
            
            return df
            
        except Exception as e:
            print(f"Error creating lag features: {e}")
            return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between different variables"""
        try:
            # Price-volume interactions
            df['price_volume_interaction'] = df['close_price'] * df['volume']
            df['return_volume_interaction'] = df['daily_return'] * df['volume']
            
            # Sentiment-price interactions
            df['sentiment_price_interaction'] = df['news_sentiment_score'] * df['close_price']
            df['sentiment_return_interaction'] = df['news_sentiment_score'] * df['daily_return']
            
            # Technical indicator interactions
            if 'rsi' in df.columns and 'volatility' in df.columns:
                df['rsi_volatility_interaction'] = df['rsi'] * df['volatility']
            
            return df
            
        except Exception as e:
            print(f"Error creating interaction features: {e}")
            return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series([50] * len(prices), index=prices.index)
    
    def get_feature_importance(self, model) -> Dict[str, float]:
        """Get feature importance from trained model"""
        try:
            if hasattr(model, 'feature_importances_'):
                feature_names = getattr(model, 'feature_names_', [])
                importances = model.feature_importances_
                
                if len(feature_names) == len(importances):
                    return dict(zip(feature_names, importances))
            
            return {}
        except Exception as e:
            print(f"Error getting feature importance: {e}")
            return {}
    
    def select_features(self, df: pd.DataFrame, method: str = 'correlation') -> List[str]:
        """Select most relevant features"""
        try:
            if method == 'correlation':
                # Select features with highest correlation to target
                target = 'close_price'
                if target in df.columns:
                    correlations = df.corr()[target].abs().sort_values(ascending=False)
                    return correlations.head(20).index.tolist()
            
            return df.columns.tolist()
            
        except Exception as e:
            print(f"Error selecting features: {e}")
            return df.columns.tolist()
