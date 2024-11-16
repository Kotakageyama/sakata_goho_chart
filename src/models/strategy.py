"""
Enhanced Transformer-based trading strategy implementation.
"""
from typing import Tuple
import numpy as np
import pandas as pd
from backtesting import Strategy
from backtesting.lib import crossover
from sklearn.cluster import KMeans

class TransformerStrategy(Strategy):
    """
    Enhanced Transformer-based trading strategy with dynamic risk management
    and market regime detection.
    """
    min_confidence = 0.6
    position_size = 0.2
    max_drawdown = 0.2
    rsi_window = 14
    atr_window = 14
    take_profit = 0.03
    stop_loss = 0.02

    def init(self):
        """Initialize strategy parameters and indicators."""
        # Initialize predictions as numpy arrays
        self.price_predictions = np.full(len(self.data.Close), np.nan)
        self.direction_predictions = np.full(len(self.data.Close), np.nan)

        # Technical indicators (already added by data loader)
        self.rsi = self.I(lambda: self.data.RSI)
        self.atr = self.I(lambda: self.data.ATR)
        self.sma_fast = self.I(lambda: self.data.SMA_10)
        self.sma_slow = self.I(lambda: self.data.SMA_20)
        self.macd = self.I(lambda: self.data.MACD)
        self.signal = self.I(lambda: self.data.Signal_Line)

        # Track equity for drawdown calculation
        self.equity_peak = self.equity

    def set_predictions(self, price_pred: np.ndarray, direction_pred: np.ndarray):
        """Set predictions for the strategy."""
        if len(price_pred) != len(self.data.Close) or len(direction_pred) != len(self.data.Close):
            raise ValueError("Prediction arrays must match data length")
        self.price_predictions = price_pred
        self.direction_predictions = direction_pred

    def should_trade(self) -> bool:
        """Determine if trading conditions are met."""
        # Check if we have valid predictions
        if np.isnan(self.price_predictions[-1]) or np.isnan(self.direction_predictions[-1]):
            return False

        # Check confidence level
        if abs(self.direction_predictions[-1] - 0.5) < self.min_confidence:
            return False

        # Check drawdown limit
        current_drawdown = (self.equity - self.equity_peak) / self.equity_peak
        if current_drawdown < -self.max_drawdown:
            return False

        # Update equity peak
        if self.equity > self.equity_peak:
            self.equity_peak = self.equity

        return True

    def next(self):
        """Execute trading logic for the current candle."""
        # Skip if trading conditions are not met
        if not self.should_trade():
            return

        # Calculate position size based on ATR
        risk_adjusted_size = self.position_size * (1 / (self.atr[-1] / self.data.Close[-1]))
        size = max(0.1, min(1.0, risk_adjusted_size))  # Limit position size between 10% and 100%

        # Calculate take-profit and stop-loss levels
        current_price = self.data.Close[-1]
        atr_multiplier = self.atr[-1] / current_price

        if self.direction_predictions[-1] > 0.5:  # Bullish signal
            if not self.position:  # Enter long position
                sl = current_price * (1 - max(self.stop_loss, 2 * atr_multiplier))
                tp = current_price * (1 + max(self.take_profit, 3 * atr_multiplier))
                self.buy(size=size, sl=sl, tp=tp)

        else:  # Bearish signal
            if not self.position:  # Enter short position
                sl = current_price * (1 + max(self.stop_loss, 2 * atr_multiplier))
                tp = current_price * (1 - max(self.take_profit, 3 * atr_multiplier))
                self.sell(size=size, sl=sl, tp=tp)

    def _calculate_atr(self) -> float:
        """Calculate ATR for risk management."""
        high = self.data.High[-self.atr_window:]
        low = self.data.Low[-self.atr_window:]
        close = self.data.Close[-self.atr_window:]

        tr1 = high - low
        tr2 = abs(high - pd.Series(close).shift())
        tr3 = abs(low - pd.Series(close).shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        return tr.mean()

    def _calculate_position_size(self, current_price: float) -> float:
        """Calculate position size based on ATR and risk parameters."""
        atr = self._calculate_atr()
        risk_per_trade = self.position_size * self.equity
        price_risk = atr * 2  # Use 2x ATR for initial stop distance

        # Calculate position size based on risk
        size = risk_per_trade / price_risk

        # Limit position size
        max_size = self.equity * 0.5 / current_price  # Maximum 50% of equity
        return min(size, max_size)
