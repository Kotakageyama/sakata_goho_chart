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
        # Store initial equity and ensure it's accessible
        self._equity = float(self.equity)  # Convert to float for safety
        assert self._equity > 0, "Initial equity must be positive"

        # Initialize equity tracking
        self.equity_peak = self._equity
        self._last_equity = self._equity

        # Convert predictions to numpy arrays for compatibility
        if not hasattr(self, 'predictions') or self.predictions is None:
            self.predictions = {
                'price': np.full(len(self.data.Close), np.nan),
                'direction': np.full(len(self.data.Close), np.nan)
            }
        else:
            # Convert to numpy arrays if they're pandas Series
            self.predictions = {
                'price': np.asarray(self.predictions['price']),
                'direction': np.asarray(self.predictions['direction'])
            }

        # Technical indicators (already added by data loader)
        self.rsi = self.I(lambda: self.data.RSI)
        self.atr = self.I(lambda: self.data.ATR)
        self.sma_fast = self.I(lambda: self.data.SMA_10)
        self.sma_slow = self.I(lambda: self.data.SMA_20)
        self.macd = self.I(lambda: self.data.MACD)
        self.signal = self.I(lambda: self.data.Signal_Line)

    def should_trade(self) -> bool:
        """Determine if trading conditions are met."""
        try:
            current_idx = len(self.data) - 1

            # Get current prediction confidence
            direction_confidence = self.predictions['direction'][current_idx]

            # Basic validation
            if np.isnan(direction_confidence):
                return False

            # Check if confidence meets minimum threshold
            if direction_confidence < self.min_confidence:
                return False

            # Check if we have enough equity
            if self._equity <= 0:
                return False

            # Check drawdown limit
            current_drawdown = (self.equity_peak - self._equity) / self.equity_peak
            if current_drawdown > self.max_drawdown:
                return False

            return True

        except Exception as e:
            print(f"Error in should_trade: {str(e)}")
            return False

    def next(self):
        """Execute trading logic."""
        try:
            if not self.should_trade():
                return

            current_idx = len(self.data) - 1
            current_price = self.data.Close[-1]

            # Update equity tracking
            self._equity = self.equity
            if self._equity > self.equity_peak:
                self.equity_peak = self._equity

            # Calculate position size
            size = self._calculate_position_size(current_price)

            # Get trading direction
            direction_confidence = self.predictions['direction'][current_idx]
            is_long = direction_confidence > 0.5

            # Place orders with proper position sizing
            if is_long and not self.position:
                self.buy(size=size, tp=current_price * (1 + self.take_profit),
                        sl=current_price * (1 - self.stop_loss))
            elif not is_long and not self.position:
                self.sell(size=size, tp=current_price * (1 - self.take_profit),
                         sl=current_price * (1 + self.stop_loss))

        except Exception as e:
            print(f"Error in next: {str(e)}")

    def _calculate_position_size(self, current_price: float) -> float:
        """Calculate position size based on ATR and risk parameters."""
        try:
            current_idx = len(self.data) - 1

            # Get ATR or use default volatility
            atr = self.atr[current_idx]
            if np.isnan(atr) or atr <= 0:
                atr = current_price * 0.02  # Default to 2% volatility

            # Calculate risk amount (fixed percentage of equity)
            risk_amount = self._equity * self.position_size

            # Ensure minimum position size
            min_position = max(0.01, self._equity * 0.01)  # At least 1% of equity

            # Calculate size based on ATR and stop loss
            stop_distance = max(atr * 2, current_price * self.stop_loss)
            size = risk_amount / stop_distance

            # Ensure size is within valid range
            max_position = self._equity / current_price  # Maximum 100% of equity
            size = max(min_position, min(size, max_position))

            return float(size)  # Ensure we return a float

        except Exception as e:
            print(f"Error in position sizing: {str(e)}")
            return 0.01  # Return minimum size on error
