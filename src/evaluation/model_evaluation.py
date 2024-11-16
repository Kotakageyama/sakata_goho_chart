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

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI technical indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR)."""
        high = data['High']
        low = data['Low']
        close = data['Close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def cross_validate(
        self,
        model: CryptoTransformer,
        strategy_class: type,
        n_splits: int = 5
    ) -> dict:
        """
        Perform cross-validation with the model and strategy.
        """
        from sklearn.model_selection import TimeSeriesSplit
        import pandas as pd

        cv_metrics = []
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Load and prepare data
        data = pd.read_csv('data/ETHUSD_2Year_2022-11-15_2024-11-15.csv')
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)

        # Prepare sequences for the entire dataset
        X, y = self.prepare_sequences(data)

        for train_idx, val_idx in tscv.split(X):
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

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
            predictions = model.predict(X_val)
            price_predictions = predictions['price_output']
            direction_predictions = predictions['direction_output']

            # Get corresponding validation data for backtesting
            val_dates = data.index[val_idx[-len(y_val):]]
            val_data = data.loc[val_dates].copy()

            # Add predictions to validation data
            val_data['price_pred'] = price_predictions
            val_data['direction_pred'] = direction_predictions

            # Calculate technical indicators
            val_data['SMA_10'] = val_data['Close'].rolling(window=10).mean()
            val_data['SMA_20'] = val_data['Close'].rolling(window=20).mean()
            val_data['RSI'] = self._calculate_rsi(val_data['Close'])
            val_data['ATR'] = self._calculate_atr(val_data[['High', 'Low', 'Close']])
            val_data['volatility'] = val_data['Close'].pct_change().rolling(window=20).std()

            # Forward fill any NaN values from indicators
            val_data.fillna(method='ffill', inplace=True)

            # Run backtest
            result = self.run_backtest(strategy_class, val_data)

            # Calculate metrics
            metrics = self.calculate_metrics(
                y_val,
                price_predictions,
                result._equity_curve['Equity'].values,
                result._equity_curve['Returns'].values
            )
            cv_metrics.append(metrics)

        # Average metrics across folds
        avg_metrics = {
            k: np.mean([m[k] for m in cv_metrics])
            for k in cv_metrics[0].keys()
        }

        return avg_metrics

    def run_backtest(self, strategy_class, data: pd.DataFrame) -> object:
        """
        Run backtesting on the given strategy and data.

        Args:
            strategy_class: The strategy class to use
            data: DataFrame containing OHLCV data and predictions

        Returns:
            Backtesting.Backtest result object
        """
        from backtesting import Backtest

        # Ensure data has required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'price_pred', 'direction_pred',
                          'SMA_10', 'SMA_20', 'RSI', 'ATR', 'volatility']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Run backtest
        bt = Backtest(data, strategy_class, cash=100000, commission=0.001)
        return bt.run()

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
