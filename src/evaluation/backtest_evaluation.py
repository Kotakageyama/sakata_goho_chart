"""
Backtesting evaluation module for the TransformerStrategy.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import optuna
from backtesting import Backtest
from src.models.strategy import TransformerStrategy

class BacktestEvaluator:
    """Evaluator class for backtesting the TransformerStrategy."""

    def __init__(self, data: pd.DataFrame):
        """Initialize with either data or file path."""
        self.data = data.copy()
        self.data.index = pd.to_datetime(self.data.index)
        self.data = self.data.sort_index()  # Ensure data is sorted
        self.predictions = None
        self._validate_data()

    def _validate_data(self):
        """Validate the input data."""
        required_columns = ['Open', 'High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        # Add Volume column if not present (optional)
        if 'Volume' not in self.data.columns:
            self.data['Volume'] = 0  # Use zero as placeholder for missing volume data

    def set_predictions(self, predictions: Dict[str, np.ndarray]):
        """Set predictions for backtesting."""
        if len(predictions['price']) != len(self.data) or len(predictions['direction']) != len(self.data):
            raise ValueError("Prediction arrays must match data length")
        self.predictions = {k: np.asarray(v) for k, v in predictions.items()}  # Ensure numpy arrays

    def _run_backtest(self, **kwargs) -> Dict:
        """Run a single backtest with given parameters."""
        if self.predictions is None:
            raise ValueError("Predictions must be set before running backtest")

        try:
            # Initialize strategy with parameters
            strategy_params = {
                'predictions': self.predictions,
                **kwargs
            }

            # Set initial cash/equity
            cash = 100000  # Start with 100k initial capital

            # Run backtest with proper initialization
            bt = Backtest(
                self.data,
                TransformerStrategy,
                cash=cash,
                commission=.002,
                exclusive_orders=True,
                trade_on_close=False,
            )

            # Run optimization
            stats = bt.run(**strategy_params)

            # Extract key metrics
            metrics = {
                'sharpe_ratio': float(stats['Sharpe Ratio']) if 'Sharpe Ratio' in stats else 0.0,
                'max_drawdown': float(stats['Max. Drawdown']) if 'Max. Drawdown' in stats else 1.0,
                'total_return': float(stats['Return [%]']) / 100 if 'Return [%]' in stats else -1.0,
                'win_rate': float(stats['Win Rate [%]']) / 100 if 'Win Rate [%]' in stats else 0.0,
                'equity_curve': stats._equity_curve['Equity'] if hasattr(stats, '_equity_curve') else pd.Series(index=self.data.index)
            }

            return metrics

        except Exception as e:
            print(f"Backtest failed: {str(e)}")
            # Return default metrics on failure
            return {
                'sharpe_ratio': 0.0,
                'max_drawdown': 1.0,
                'total_return': -1.0,
                'win_rate': 0.0,
                'equity_curve': pd.Series(index=self.data.index)
            }

    def optimize_hyperparameters(self, n_trials: int = 30) -> Dict:
        """Optimize strategy hyperparameters using Optuna."""
        def objective(trial):
            params = {
                'min_confidence': trial.suggest_float('min_confidence', 0.5, 0.8),
                'position_size': trial.suggest_float('position_size', 0.1, 0.4),
                'max_drawdown': trial.suggest_float('max_drawdown', 0.1, 0.3),
                'rsi_window': trial.suggest_int('rsi_window', 10, 30),
                'atr_window': trial.suggest_int('atr_window', 10, 30),
                'take_profit': trial.suggest_float('take_profit', 0.01, 0.05),
                'stop_loss': trial.suggest_float('stop_loss', 0.01, 0.05)
            }

            results = self._run_backtest(**params)
            score = results['sharpe_ratio'] * (1 - results['max_drawdown'])
            return score if not np.isnan(score) else 0.0

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        return {
            'best_params': study.best_params,
            'best_value': study.best_value
        }

    def run_cross_validation(self, params: Dict, n_splits: int = 5) -> Dict[str, List[float]]:
        """Perform time series cross-validation."""
        results = {
            'sharpe_ratio': [],
            'max_drawdown': [],
            'total_return': [],
            'win_rate': []
        }

        # Split data into n_splits segments
        segment_size = len(self.data) // n_splits
        for i in range(n_splits):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size if i < n_splits - 1 else len(self.data)

            # Create segment data and predictions
            segment_data = self.data.iloc[start_idx:end_idx].copy()
            segment_predictions = {
                'price': self.predictions['price'][start_idx:end_idx],
                'direction': self.predictions['direction'][start_idx:end_idx]
            }

            # Create temporary evaluator for segment
            temp_evaluator = BacktestEvaluator(segment_data)
            temp_evaluator.set_predictions(segment_predictions)

            # Run backtest on segment
            segment_results = temp_evaluator._run_backtest(**params)

            # Store results
            for metric in results.keys():
                results[metric].append(segment_results[metric])

        return results

    def generate_evaluation_report(self) -> Dict:
        """Generate comprehensive evaluation report."""
        # Optimize hyperparameters
        optimization_results = self.optimize_hyperparameters()

        # Run cross-validation with best parameters
        cv_results = self.run_cross_validation(optimization_results['best_params'])

        # Run final backtest with best parameters
        final_results = self._run_backtest(**optimization_results['best_params'])

        return {
            'optimization_results': optimization_results,
            'cross_validation_results': cv_results,
            'final_backtest_results': final_results
        }
