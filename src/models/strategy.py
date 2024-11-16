"""
TransformerStrategy implementation for cryptocurrency trading.
"""
import numpy as np
import pandas as pd
from backtesting import Strategy
from typing import Dict, Optional

class TransformerStrategy(Strategy):
    """Trading strategy using Transformer model predictions."""

    # Class parameters for optimization
    min_confidence = 0.52  # Lowered from 0.6 to allow more trades
    position_size = 0.2
    max_drawdown = 0.2
    rsi_window = 14
    atr_window = 14
    take_profit = 0.03
    stop_loss = 0.02
    predictions: Optional[Dict[str, pd.Series]] = None

    def init(self):
        """Initialize strategy parameters and indicators."""
        # Ensure predictions are set
        if not hasattr(self, 'predictions') or self.predictions is None:
            self.predictions = {
                'price': pd.Series(np.full(len(self.data.Close), np.nan), index=self.data.Close.index),
                'direction': pd.Series(np.full(len(self.data.Close), np.nan), index=self.data.Close.index)
            }

        # Technical indicators (already added by data loader)
        self.rsi = self.I(lambda: self.data.RSI)
        self.atr = self.I(lambda: self.data.ATR)
        self.sma_fast = self.I(lambda: self.data.SMA_10)
        self.sma_slow = self.I(lambda: self.data.SMA_20)
        self.macd = self.I(lambda: self.data.MACD)
        self.signal = self.I(lambda: self.data.Signal_Line)

        # Track equity for drawdown calculation
        self.equity_peak = self.equity

    def should_trade(self) -> bool:
        """Determine if trading conditions are met."""
        current_idx = len(self.data) - 1

        # Check if we have valid predictions
        if np.isnan(self.predictions['direction'].iloc[current_idx]):
            return False

        # Check confidence level (less restrictive)
        confidence = abs(self.predictions['direction'].iloc[current_idx] - 0.5)
        if confidence < (self.min_confidence - 0.5):  # Adjusted threshold calculation
            return False

        # Check drawdown limit
        current_drawdown = (self.equity - self.equity_peak) / self.equity_peak
        if current_drawdown < -self.max_drawdown:
            return False

        # Update equity peak
        if self.equity > self.equity_peak:
            self.equity_peak = self.equity

        # Additional trading conditions
        if self.position:  # If we have a position, be more conservative
            return False

        return True

    def next(self):
        """Execute trading logic for the current candle."""
        current_idx = len(self.data) - 1

        # Skip if trading conditions are not met
        if not self.should_trade():
            return

        # Calculate position size based on ATR
        current_price = self.data.Close.iloc[current_idx]
        size = self._calculate_position_size(current_price)

        # Calculate take-profit and stop-loss levels
        atr_multiplier = self.atr.iloc[current_idx] / current_price

        if self.predictions['direction'].iloc[current_idx] > 0.5:  # Bullish signal
            if not self.position:  # Enter long position
                sl = current_price * (1 - max(self.stop_loss, 2 * atr_multiplier))
                tp = current_price * (1 + max(self.take_profit, 3 * atr_multiplier))
                self.buy(size=size, sl=sl, tp=tp)

        else:  # Bearish signal
            if not self.position:  # Enter short position
                sl = current_price * (1 + max(self.stop_loss, 2 * atr_multiplier))
                tp = current_price * (1 - max(self.take_profit, 3 * atr_multiplier))
                self.sell(size=size, sl=sl, tp=tp)

    def _calculate_position_size(self, current_price: float) -> float:
        """Calculate position size based on ATR and risk parameters."""
        current_idx = len(self.data) - 1
        atr = self.atr.iloc[current_idx]
        risk_per_trade = self.position_size * self.equity
        price_risk = atr * 2  # Use 2x ATR for initial stop distance

        # Calculate position size based on risk
        size = risk_per_trade / (current_price * self.stop_loss)

        # Limit position size
        max_size = self.equity * 0.5 / current_price  # Maximum 50% of equity
        return min(size, max_size)
