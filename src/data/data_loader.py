"""
Data loader module for cryptocurrency price data.
Supports both local CSV files and Google Drive paths.
"""
import pandas as pd
import numpy as np
from typing import List, Union, Optional
import ta
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess data from a CSV file.
    Adds technical indicators and handles missing values.

    Args:
        file_path: Path to the CSV file

    Returns:
        Preprocessed DataFrame with technical indicators
    """
    loader = CryptoDataLoader()
    # Load raw data
    df = loader.load_data(file_path)
    # Add technical indicators
    df = loader.add_technical_indicators(df)
    # Handle missing values
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df

class CryptoDataLoader:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_10', 'SMA_20', 'RSI', 'MACD', 'Signal_Line',
            'ATR', 'BB_upper', 'BB_middle', 'BB_lower',
            'Stoch_K', 'Stoch_D', 'ADX'
        ]

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from either local CSV or Google Drive path
        """
        try:
            if 'drive/MyDrive' in file_path:
                # For Google Colab environment
                from google.colab import drive
                drive.mount('/content/drive')

            # Expand user path if necessary
            file_path = file_path.replace('~', str(Path.home()))

            df = pd.read_csv(file_path)

            # Ensure datetime format
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

            return df
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the dataset
        """
        # Price-based indicators
        df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['Signal_Line'] = macd.macd_signal()

        # Volatility indicators
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['BB_upper'] = bollinger.bollinger_hband()
        df['BB_middle'] = bollinger.bollinger_mavg()
        df['BB_lower'] = bollinger.bollinger_lband()

        # Momentum indicators
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()

        # Trend indicators
        df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'])

        return df

    def prepare_data(
        self,
        df: pd.DataFrame,
        sequence_length: int = 60,
        target_column: str = 'Close',
        train_split: float = 0.8
    ) -> tuple:
        """
        Prepare data for model training
        """
        # Add technical indicators
        df = self.add_technical_indicators(df)

        # Forward fill NaN values
        df = df.fillna(method='ffill')

        # Drop remaining NaN values
        df = df.dropna()

        # Scale features
        scaled_data = self.scaler.fit_transform(df[self.feature_columns])
        scaled_df = pd.DataFrame(scaled_data, columns=self.feature_columns, index=df.index)

        # Create sequences
        X, y = [], []
        for i in range(len(scaled_df) - sequence_length):
            X.append(scaled_df.iloc[i:(i + sequence_length)].values)
            y.append(scaled_df[target_column].iloc[i + sequence_length])

        X = np.array(X)
        y = np.array(y)

        # Split data
        train_size = int(len(X) * train_split)
        X_train = X[:train_size]
        X_test = X[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]

        return X_train, X_test, y_train, y_test, self.scaler

    def inverse_transform_predictions(
        self,
        predictions: np.ndarray,
        feature_index: int
    ) -> np.ndarray:
        """
        Inverse transform scaled predictions
        """
        dummy = np.zeros((len(predictions), len(self.feature_columns)))
        dummy[:, feature_index] = predictions
        return self.scaler.inverse_transform(dummy)[:, feature_index]
