"""
Backtesting evaluation module for the TransformerStrategy.
"""
from typing import Dict, List, Union
import numpy as np
import pandas as pd
import optuna
from backtesting import Backtest
from sklearn.model_selection import TimeSeriesSplit
from ..data.data_loader import CryptoDataLoader
from ..models.strategy import TransformerStrategy

class BacktestEvaluator:
    """Evaluator class for backtesting the TransformerStrategy."""

    def __init__(self, data: Union[str, pd.DataFrame]):
        """Initialize the evaluator with data."""
        self.data_loader = CryptoDataLoader()
        if isinstance(data, str):
            self.data = self.data_loader.load_data(data)
        else:
            self.data = data.copy()

        # Ensure data is sorted
        if isinstance(self.data.index, pd.DatetimeIndex):
            self.data.sort_index(inplace=True)

        self.predictions = {
            'price': np.full(len(self.data), np.nan),
            'direction': np.full(len(self.data), np.nan)
        }

    def set_predictions(self, predictions: Dict[str, np.ndarray]):
        """Set predictions for evaluation."""
        if len(predictions['price']) != len(self.data) or len(predictions['direction']) != len(self.data):
            raise ValueError("Prediction arrays must match data length")
        self.predictions = predictions

    def calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio with proper error handling."""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        try:
            sharpe = np.sqrt(252) * (returns.mean() / returns.std())
            return 0.0 if np.isnan(sharpe) else sharpe
        except:
            return 0.0

    def calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown with proper error handling."""
        try:
            peak = equity_curve.expanding(min_periods=1).max()
            drawdown = (equity_curve - peak) / peak
            max_dd = abs(drawdown.min())
            return 0.0 if np.isnan(max_dd) else max_dd
        except:
            return 0.0

    def run_backtest(self, cash: float = 100000, commission: float = 0.001,
                    **strategy_params) -> Dict:
        """Run backtest with current settings."""
        try:
            strategy_params['predictions'] = self.predictions
            bt = Backtest(
                self.data,
                TransformerStrategy,
                cash=cash,
                commission=commission,
                exclusive_orders=True
            )
            stats = bt.run(**strategy_params)

            # Calculate additional metrics
            returns = pd.Series(stats._equity_curve['Equity'].pct_change().dropna())
            sharpe = self.calculate_sharpe_ratio(returns)
            max_dd = self.calculate_max_drawdown(stats._equity_curve['Equity'])

            return {
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'total_return': stats['Return [%]'] / 100,
                'win_rate': stats['Win Rate [%]'] / 100,
                'equity_curve': stats._equity_curve['Equity']
            }
        except Exception as e:
            print(f"Backtest failed: {str(e)}")
            return {
                'sharpe_ratio': 0.0,
                'max_drawdown': 1.0,
                'total_return': -1.0,
                'win_rate': 0.0,
                'equity_curve': pd.Series([])
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

            results = self.run_backtest(**params)
            score = results['sharpe_ratio'] * (1 - results['max_drawdown'])

            # Penalize invalid results
            if np.isnan(score) or np.isinf(score):
                return -1.0

            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        return {
            'best_params': study.best_params if study.trials_dataframe is not None else {},
            'best_value': study.best_value if study.trials_dataframe is not None else 0.0
        }

    def run_cross_validation(self, n_splits: int = 5) -> Dict[str, List[float]]:
        """Run time series cross-validation."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_results = {
            'sharpe_ratio': [],
            'max_drawdown': [],
            'total_return': [],
            'win_rate': []
        }

        for train_idx, test_idx in tscv.split(self.data):
            try:
                # Prepare data for this fold
                test_data = self.data.iloc[test_idx]
                test_predictions = {
                    'price': self.predictions['price'][test_idx],
                    'direction': self.predictions['direction'][test_idx]
                }

                # Create and run evaluator for this fold
                fold_evaluator = BacktestEvaluator(test_data)
                fold_evaluator.set_predictions(test_predictions)
                results = fold_evaluator.run_backtest()

                # Store results
                for metric in cv_results:
                    cv_results[metric].append(results[metric])
            except Exception as e:
                print(f"Cross-validation fold failed: {str(e)}")
                continue

        return cv_results

    def generate_evaluation_report(self) -> Dict:
        """Generate comprehensive evaluation report."""
        try:
            # Run optimization
            opt_results = self.optimize_hyperparameters(n_trials=30)

            # Run cross-validation
            cv_results = self.run_cross_validation()

            # Run final backtest with optimized parameters
            final_results = self.run_backtest(**opt_results['best_params'])

            return {
                'optimization_results': opt_results,
                'cross_validation_results': cv_results,
                'final_backtest_results': final_results
            }
        except Exception as e:
            print(f"Error generating evaluation report: {str(e)}")
            return {
                'error': str(e),
                'optimization_results': {},
                'cross_validation_results': {},
                'final_backtest_results': {}
            }
