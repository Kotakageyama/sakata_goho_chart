"""
Model evaluation and tuning module for the CryptoTransformer model.
Includes cross-validation, hyperparameter optimization, and performance metrics.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, accuracy_score
from typing import Dict, List, Tuple, Optional
import optuna
from backtesting import Backtest
import tensorflow as tf
from ..models.transformer_model import CryptoTransformer
from ..models.strategy import TransformerStrategy

class ModelEvaluator:
    """
    Comprehensive model evaluation and tuning class.
    """
    def __init__(
        self,
        data: pd.DataFrame,
        validation_split: float = 0.2,
        n_splits: int = 5,
        random_seed: int = 42
    ):
        self.data = data
        self.validation_split = validation_split
        self.n_splits = n_splits
        self.random_seed = random_seed
        self.performance_metrics = {}

    def calculate_sharpe_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.02
    ) -> float:
        """
        Calculate the Sharpe ratio of the strategy.
        """
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        if len(excess_returns) < 2:
            return 0.0
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)

    def calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """
        Calculate the maximum drawdown from peak equity.
        """
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        return np.min(drawdown)

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        equity_curve: np.ndarray,
        returns: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        """
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'sharpe_ratio': self.calculate_sharpe_ratio(returns),
            'max_drawdown': self.calculate_max_drawdown(equity_curve),
            'total_return': (equity_curve[-1] / equity_curve[0]) - 1,
            'volatility': np.std(returns) * np.sqrt(252),
            'win_rate': np.mean(returns > 0)
        }
        return metrics

    def cross_validate(
        self,
        model: CryptoTransformer,
        strategy: TransformerStrategy
    ) -> Dict[str, List[float]]:
        """
        Perform time series cross-validation.
        """
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        cv_metrics = {
            'rmse': [], 'sharpe_ratio': [], 'max_drawdown': [],
            'total_return': [], 'volatility': [], 'win_rate': []
        }

        for train_idx, val_idx in tscv.split(self.data):
            # Split data
            train_data = self.data.iloc[train_idx]
            val_data = self.data.iloc[val_idx]

            # Train model
            model.fit(train_data)

            # Generate predictions
            predictions = model.predict(val_data)

            # Run backtest
            bt = Backtest(val_data, strategy, cash=100000, commission=0.001)
            result = bt.run()

            # Calculate metrics
            metrics = self.calculate_metrics(
                val_data['Close'].values,
                predictions,
                result._equity_curve['Equity'].values,
                result._equity_curve['Returns'].values
            )

            # Store metrics
            for key, value in metrics.items():
                cv_metrics[key].append(value)

        return cv_metrics

    def optimize_hyperparameters(
        self,
        n_trials: int = 100
    ) -> Dict[str, float]:
        """
        Optimize model hyperparameters using Optuna.
        """
        def objective(trial):
            # Define hyperparameter search space
            params = {
                'n_heads': trial.suggest_int('n_heads', 4, 12),
                'n_layers': trial.suggest_int('n_layers', 2, 6),
                'd_model': trial.suggest_int('d_model', 32, 128, step=32),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
            }

            # Create and train model with suggested parameters
            model = CryptoTransformer(**params)
            cv_metrics = self.cross_validate(model, TransformerStrategy)

            # Use Sharpe ratio as optimization metric
            return np.mean(cv_metrics['sharpe_ratio'])

        # Create study and optimize
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        return study.best_params

    def compare_models(
        self,
        old_model: CryptoTransformer,
        new_model: CryptoTransformer
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare performance between old and new models.
        """
        models = {'old': old_model, 'new': new_model}
        comparison = {}

        for name, model in models.items():
            cv_metrics = self.cross_validate(model, TransformerStrategy)
            comparison[name] = {
                key: np.mean(values) for key, values in cv_metrics.items()
            }

        return comparison

    def generate_report(self) -> str:
        """
        Generate a comprehensive evaluation report.
        """
        report = []
        report.append("# モデル評価レポート\n")

        # Add cross-validation results
        report.append("## クロスバリデーション結果")
        for metric, values in self.performance_metrics.items():
            mean_value = np.mean(values)
            std_value = np.std(values)
            report.append(f"\n{metric}:")
            report.append(f"  平均: {mean_value:.4f}")
            report.append(f"  標準偏差: {std_value:.4f}")

        # Add model comparison if available
        if hasattr(self, 'model_comparison'):
            report.append("\n## モデル比較")
            for model_name, metrics in self.model_comparison.items():
                report.append(f"\n{model_name}モデル:")
                for metric, value in metrics.items():
                    report.append(f"  {metric}: {value:.4f}")

        return "\n".join(report)
