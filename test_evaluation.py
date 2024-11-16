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

# Generate dummy predictions
n_samples = len(data)
price_pred = np.random.normal(data['Close'].mean(), data['Close'].std(), n_samples)
direction_pred = np.random.uniform(0, 1, n_samples)

# Run evaluation
evaluator = BacktestEvaluator(data)
evaluator.set_predictions({'price': price_pred, 'direction': direction_pred})
report = evaluator.generate_evaluation_report()
print('\nEvaluation Report:\n', report)
