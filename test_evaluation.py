"""
Test script for evaluating the TransformerStrategy.
"""
import numpy as np
import pandas as pd
from src.evaluation.backtest_evaluation import BacktestEvaluator
from src.data.data_loader import load_and_preprocess_data

def generate_test_predictions(data: pd.DataFrame) -> dict:
    """Generate test predictions for backtesting."""
    # Calculate technical indicators for prediction generation
    rsi = data['RSI'].values
    macd = data['MACD'].values
    signal = data['Signal_Line'].values
    momentum = data['Close'].pct_change(5).values

    # Combine signals for direction prediction
    direction = np.zeros(len(data))

    # RSI signals (oversold/overbought)
    direction += np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0))

    # MACD signals
    direction += np.where(macd > signal, 1, -1)

    # Momentum signals
    direction += np.sign(momentum)

    # Normalize and convert to probability
    direction = direction / 3  # Scale to [-1, 1]
    direction = 1 / (1 + np.exp(-5 * direction))  # Sigmoid transform

    # Generate price predictions (simple moving average + noise)
    sma = data['SMA_20'].values
    noise = np.random.normal(0, data['ATR'].values * 0.1)
    price = sma + noise

    # Handle NaN values
    direction = np.nan_to_num(direction, nan=0.5)
    price = np.nan_to_num(price, nan=data['Close'].iloc[-1])

    return {
        'price': price,
        'direction': direction
    }

def main():
    """Main function to run the evaluation."""
    # Load and preprocess data
    data = load_and_preprocess_data('~/attachments/ETHUSD_2Year_2022-11-15_2024-11-15.csv')

    # Initialize evaluator
    evaluator = BacktestEvaluator(data)

    # Generate and set predictions
    predictions = generate_test_predictions(data)
    evaluator.set_predictions(predictions)

    # Generate evaluation report
    report = evaluator.generate_evaluation_report()

    # Print evaluation results
    print("\nEvaluation Report:\n", report)

    # Print data statistics
    print("\nData Statistics:")
    print(f"Total samples: {len(data)}")
    print(f"Direction predictions > 0.5: {np.sum(predictions['direction'] > 0.5)}")
    print(f"Direction predictions < 0.5: {np.sum(predictions['direction'] < 0.5)}")
    print(f"Average direction confidence: {np.mean(np.abs(predictions['direction'] - 0.5)):.4f}")
    print(f"Data range: {data.index[0]} to {data.index[-1]}")

if __name__ == "__main__":
    main()
