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
            self.data = data
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
        """Calculate Sharpe ratio."""
        if len(returns) == 0:
            return 0.0
        return np.sqrt(252) * (returns.mean() / returns.std())

    def calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown."""
        peak = equity_curve.expanding(min_periods=1).max()
        drawdown = (equity_curve - peak) / peak
        return abs(drawdown.min())

    def run_backtest(self, cash: float = 100000, commission: float = 0.001) -> Dict:
        """Run backtest with current settings."""
        # Create strategy instance
        strategy = TransformerStrategy
        strategy.price_predictions = self.predictions['price']
        strategy.direction_predictions = self.predictions['direction']

        # Run backtest
        bt = Backtest(
            self.data,
            strategy,
            cash=cash,
            commission=commission,
            exclusive_orders=True
        )

        stats = bt.run()

        # Calculate additional metrics
        returns = pd.Series(stats._equity_curve['Equity']).pct_change()
        sharpe_ratio = self.calculate_sharpe_ratio(returns.dropna())
        max_drawdown = self.calculate_max_drawdown(pd.Series(stats._equity_curve['Equity']))

        return {
            'stats': stats,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_return': stats['Return [%]'],
            'win_rate': stats['Win Rate [%]'],
            'profit_factor': stats['Profit Factor']
        }

    def optimize_hyperparameters(self, n_trials: int = 50) -> Dict:
        """Optimize strategy hyperparameters using Optuna."""
        def objective(trial):
            # Define hyperparameter space
            params = {
                'min_confidence': trial.suggest_float('min_confidence', 0.5, 0.8),
                'position_size': trial.suggest_float('position_size', 0.1, 0.5),
                'max_drawdown': trial.suggest_float('max_drawdown', 0.1, 0.3),
                'rsi_window': trial.suggest_int('rsi_window', 10, 30),
                'atr_window': trial.suggest_int('atr_window', 10, 30),
                'take_profit': trial.suggest_float('take_profit', 0.01, 0.05),
                'stop_loss': trial.suggest_float('stop_loss', 0.01, 0.05)
            }

            # Create strategy instance with parameters
            strategy = TransformerStrategy
            for key, value in params.items():
                setattr(strategy, key, value)
            strategy.price_predictions = self.predictions['price']
            strategy.direction_predictions = self.predictions['direction']

            # Run backtest
            bt = Backtest(
                self.data,
                strategy,
                cash=100000,
                commission=0.001,
                exclusive_orders=True
            )

            stats = bt.run()

            # Calculate objective value (combination of metrics)
            returns = pd.Series(stats._equity_curve['Equity']).pct_change()
            sharpe_ratio = self.calculate_sharpe_ratio(returns.dropna())
            max_drawdown = self.calculate_max_drawdown(pd.Series(stats._equity_curve['Equity']))

            # Objective: maximize Sharpe ratio while minimizing drawdown
            return sharpe_ratio * (1 + abs(max_drawdown))

        # Create and run optimization study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        return {
            'best_params': study.best_params,
            'best_value': study.best_value
        }

    def run_cross_validation(self, n_splits: int = 5) -> List[Dict]:
        """Perform time series cross-validation."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_results = []

        for train_idx, test_idx in tscv.split(self.data):
            # Split data
            test_data = self.data.iloc[test_idx]
            test_predictions = {
                'price': self.predictions['price'][test_idx],
                'direction': self.predictions['direction'][test_idx]
            }

            # Create strategy instance
            strategy = TransformerStrategy
            strategy.price_predictions = test_predictions['price']
            strategy.direction_predictions = test_predictions['direction']

            # Run backtest on test set
            bt = Backtest(
                test_data,
                strategy,
                cash=100000,
                commission=0.001,
                exclusive_orders=True
            )

            stats = bt.run()

            # Calculate metrics
            returns = pd.Series(stats._equity_curve['Equity']).pct_change()
            cv_results.append({
                'sharpe_ratio': self.calculate_sharpe_ratio(returns.dropna()),
                'max_drawdown': self.calculate_max_drawdown(pd.Series(stats._equity_curve['Equity'])),
                'total_return': stats['Return [%]'],
                'win_rate': stats['Win Rate [%]']
            })

        return cv_results

    def generate_evaluation_report(self) -> str:
        """Generate a comprehensive evaluation report."""
        # Run basic backtest
        basic_results = self.run_backtest()

        # Run cross-validation
        cv_results = self.run_cross_validation()

        # Run hyperparameter optimization
        opt_results = self.optimize_hyperparameters(n_trials=30)

        # Calculate average metrics from cross-validation
        avg_cv_metrics = pd.DataFrame(cv_results).mean()

        report = f"""
        Transformer Strategy Evaluation Report
        ====================================

        Basic Backtest Results:
        ----------------------
        Total Return: {basic_results['total_return']:.2f}%
        Sharpe Ratio: {basic_results['sharpe_ratio']:.2f}
        Maximum Drawdown: {basic_results['max_drawdown']:.2f}
        Win Rate: {basic_results['win_rate']:.2f}%
        Profit Factor: {basic_results['profit_factor']:.2f}

        Cross-validation Results (Average):
        ---------------------------------
        Sharpe Ratio: {avg_cv_metrics['sharpe_ratio']:.2f}
        Maximum Drawdown: {avg_cv_metrics['max_drawdown']:.2f}
        Total Return: {avg_cv_metrics['total_return']:.2f}%
        Win Rate: {avg_cv_metrics['win_rate']:.2f}%

        Optimized Hyperparameters:
        -------------------------
        {opt_results['best_params']}
        Best Sharpe Ratio: {opt_results['best_value']:.2f}
        """

        return report

def main():
    """Main function to run the evaluation."""
    # Initialize evaluator with data
    data = pd.read_csv('data/ETHUSD_2Year_2022-11-15_2024-11-15.csv')
    evaluator = BacktestEvaluator(data)

    # Generate dummy predictions for testing (replace with actual model predictions)
    n_samples = len(evaluator.data)
    evaluator.set_predictions(
        price_pred=np.random.randn(n_samples),
        direction_pred=np.random.rand(n_samples)
    )

    # Generate and print evaluation report
    report = evaluator.generate_evaluation_report()
    print(report)

if __name__ == '__main__':
    main()
