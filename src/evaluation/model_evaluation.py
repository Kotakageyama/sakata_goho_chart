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
from ..models.transformer_model import CryptoTransformer, create_model
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
        sequence_length: int = 60,
        random_seed: int = 42
    ):
        self.data = data
        self.validation_split = validation_split
        self.n_splits = n_splits
        self.sequence_length = sequence_length
        self.random_seed = random_seed
        self.performance_metrics = {}

    def prepare_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for the transformer model.
        """
        features = ['Open', 'High', 'Low', 'Close']  # Removed Volume as it's not in the data
        sequences = []
        targets = []

        for i in range(len(data) - self.sequence_length):
            seq = data[features].iloc[i:i+self.sequence_length].values
            target = data['Close'].iloc[i+self.sequence_length]
            sequences.append(seq)
            targets.append(target)

        return np.array(sequences), np.array(targets)

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
        strategy_class: type
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

            # Prepare sequences
            X_train, y_train = self.prepare_sequences(train_data)
            X_val, y_val = self.prepare_sequences(val_data)

            # Train model
            model.fit(
                X_train,
                {'price_output': y_train, 'direction_output': np.sign(np.diff(y_train, prepend=y_train[0]))},
                validation_split=0.2,
                epochs=50,
                batch_size=32,
                verbose=0
            )

            # Generate predictions
            predictions = model.predict(X_val)[0]  # Get price predictions

            # Initialize and run strategy
            strategy = strategy_class(
                model=model,
                sequence_length=self.sequence_length
            )
            bt = Backtest(
                val_data,
                strategy,
                cash=100000,
                commission=0.001
            )
            result = bt.run()

            # Calculate metrics
            metrics = self.calculate_metrics(
                y_val,
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
                'sequence_length': self.sequence_length,
                'num_features': 4,  # OHLC (removed Volume)
                'd_model': trial.suggest_int('d_model', 32, 128, step=32),
                'num_heads': trial.suggest_int('num_heads', 4, 12),
                'ff_dim': trial.suggest_int('ff_dim', 64, 256, step=64),
                'num_transformer_blocks': trial.suggest_int('num_transformer_blocks', 2, 6),
                'mlp_units': [
                    trial.suggest_int('mlp_units_1', 32, 128, step=32),
                    trial.suggest_int('mlp_units_2', 16, 64, step=16)
                ],
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'mlp_dropout': trial.suggest_float('mlp_dropout', 0.1, 0.5)
            }

            # Create and train model with suggested parameters
            model = create_model(**params)
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
