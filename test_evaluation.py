import numpy as np
import pandas as pd
from src.evaluation.backtest_evaluation import BacktestEvaluator

# Load and prepare data
data = pd.read_csv('~/attachments/ETHUSD_2Year_2022-11-15_2024-11-15.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Add technical indicators
data['RSI'] = data['Close'].diff().rolling(window=14).apply(
    lambda x: 100 - (100 / (1 + (x[x > 0].mean() / -x[x < 0].mean()))) if len(x[x > 0]) > 0 and len(x[x < 0]) > 0 else 50
)
data['ATR'] = data['High'] - data['Low']
data['SMA_10'] = data['Close'].rolling(window=10).mean()
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['MACD'] = data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()
data['Signal_Line'] = data['MACD'].ewm(span=9).mean()

# Fill NaN values with appropriate defaults
data = data.fillna(method='ffill').fillna(method='bfill')

# Generate more realistic predictions for testing
n_samples = len(data)

# Price predictions: Use a simple moving average + noise
price_pred = data['Close'].rolling(window=20).mean().values
price_pred += np.random.normal(0, data['Close'].std() * 0.1, n_samples)

# Direction predictions: Generate based on price momentum and technical indicators
momentum = data['Close'].pct_change(periods=5).values
rsi_signal = (data['RSI'] > 50).astype(float).values
macd_signal = (data['MACD'] > data['Signal_Line']).astype(float).values

# Combine signals
direction_pred = (momentum + rsi_signal + macd_signal) / 3
direction_pred = 1 / (1 + np.exp(-5 * (direction_pred - 0.5)))  # Sigmoid transform
direction_pred = np.nan_to_num(direction_pred, nan=0.5)  # Handle NaN values

print("\nInitializing evaluator...")
evaluator = BacktestEvaluator(data)
print("Setting predictions...")
evaluator.set_predictions({
    'price': price_pred,
    'direction': direction_pred
})

print("\nRunning evaluation with debug mode...")
report = evaluator.generate_evaluation_report()
print('\nEvaluation Report:\n', report)

print("\nData Statistics:")
print(f"Total samples: {n_samples}")
print(f"Direction predictions > 0.5: {np.sum(direction_pred > 0.5)}")
print(f"Direction predictions < 0.5: {np.sum(direction_pred < 0.5)}")
print(f"Average direction confidence: {np.mean(np.abs(direction_pred - 0.5)):.4f}")
print(f"Data range: {data.index.min()} to {data.index.max()}")
