import numpy as np
import pandas as pd
from src.evaluation.backtest_evaluation import BacktestEvaluator

# Load and prepare data
data = pd.read_csv('~/attachments/ETHUSD_2Year_2022-11-15_2024-11-15.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Add technical indicators
data['RSI'] = data['Close'].diff().rolling(window=14).apply(lambda x: 100 - (100 / (1 + (x[x > 0].mean() / -x[x < 0].mean()))))
data['ATR'] = data['High'] - data['Low']
data['SMA_10'] = data['Close'].rolling(window=10).mean()
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['MACD'] = data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()
data['Signal_Line'] = data['MACD'].ewm(span=9).mean()

# Generate more realistic predictions for testing
n_samples = len(data)
# Price predictions: Use a simple moving average + noise
price_pred = data['Close'].rolling(window=20).mean().values
price_pred += np.random.normal(0, data['Close'].std() * 0.1, n_samples)

# Direction predictions: Generate based on price momentum
momentum = data['Close'].pct_change(periods=5).values
direction_pred = 1 / (1 + np.exp(-10 * momentum))  # Sigmoid transform
direction_pred = np.nan_to_num(direction_pred, nan=0.5)  # Handle NaN values

# Initialize evaluator and set predictions
evaluator = BacktestEvaluator(data)
evaluator.set_predictions({
    'price': price_pred,
    'direction': direction_pred
})

# Run evaluation with debug output
print("\nRunning evaluation with debug mode...")
report = evaluator.generate_evaluation_report()
print('\nEvaluation Report:\n', report)

# Print additional debug information
print("\nData Statistics:")
print(f"Total samples: {n_samples}")
print(f"Direction predictions > 0.5: {np.sum(direction_pred > 0.5)}")
print(f"Direction predictions < 0.5: {np.sum(direction_pred < 0.5)}")
print(f"Average direction confidence: {np.mean(np.abs(direction_pred - 0.5)):.4f}")
