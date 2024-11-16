"""
TransformerStrategy implementation for cryptocurrency trading.
"""
import numpy as np
import pandas as pd
from backtesting import Strategy
from typing import Dict, Optional, Union

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
    predictions: Optional[Dict[str, Union[np.ndarray, pd.Series]]] = None

    def init(self):
        """Initialize strategy parameters and indicators."""
        # Convert predictions to numpy arrays for compatibility
        if not hasattr(self, 'predictions') or self.predictions is None:
            self.predictions = {
                'price': np.full(len(self.data.Close), np.nan),
                'direction': np.full(len(self.data.Close), np.nan)
            }
        else:
            # Convert to numpy arrays if they're pandas Series
            self.predictions = {
                'price': self.predictions['price'].values if hasattr(self.predictions['price'], 'values')
                        else np.asarray(self.predictions['price']),
                'direction': self.predictions['direction'].values if hasattr(self.predictions['direction'], 'values')
                        else np.asarray(self.predictions['direction'])
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
        if np.isnan(self.predictions['direction'][current_idx]):
            return False

        # Check confidence level (less restrictive)
        confidence = abs(self.predictions['direction'][current_idx] - 0.5)
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
        current_price = self.data.Close[current_idx]
        size = self._calculate_position_size(current_price)

        # Calculate take-profit and stop-loss levels
        atr_multiplier = self.atr[current_idx] / current_price

        if self.predictions['direction'][current_idx] > 0.5:  # Bullish signal
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
        atr = self.atr[current_idx]

        # Ensure we have valid ATR value
        if np.isnan(atr) or atr <= 0:
            atr = current_price * 0.02  # Default to 2% volatility

        # Calculate risk amount (fixed percentage of equity)
        risk_amount = self.equity * self.position_size

        # Calculate position size based on ATR for stop loss
        stop_distance = max(atr * 2, current_price * self.stop_loss)  # Use larger of ATR or fixed stop

        # Calculate size in units
        size = risk_amount / stop_distance

        # Ensure size is positive and within limits
        size = max(0.01, min(size, self.equity / current_price))  # Minimum 1% of position, maximum 100% of equity

        return size
