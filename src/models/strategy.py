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
                print("Skipping trade: NaN direction confidence")
                return False

            # Check if confidence meets minimum threshold (less restrictive)
            confidence_threshold = 0.45  # Lower threshold for more trades
            if abs(direction_confidence - 0.5) < (self.min_confidence - confidence_threshold):
                print(f"Skipping trade: Low confidence {direction_confidence}")
                return False

            # Check if we have enough equity (more permissive)
            if self._equity < 1000:  # Minimum equity requirement
                print("Skipping trade: Insufficient equity")
                return False

            # Check drawdown limit (more permissive)
            current_drawdown = (self.equity_peak - self._equity) / self.equity_peak
            if current_drawdown > self.max_drawdown * 1.5:  # Allow 50% more drawdown
                print(f"Skipping trade: Exceeded drawdown limit {current_drawdown}")
                return False

            print(f"Trade conditions met: confidence={direction_confidence}, equity={self._equity}")
            return True

        except Exception as e:
            print(f"Error in should_trade: {str(e)}")
            return False

    def next(self):
        """Execute trading logic."""
        try:
            current_idx = len(self.data) - 1
            current_price = self.data.Close[-1]

            # Update equity tracking
            self._equity = self.equity
            if self._equity > self.equity_peak:
                self.equity_peak = self._equity

            # Check trading conditions
            if not self.should_trade():
                return

            # Calculate position size
            size = self._calculate_position_size(current_price)
            if size <= 0:
                print(f"Invalid position size: {size}")
                return

            # Get trading direction (more permissive)
            direction_confidence = self.predictions['direction'][current_idx]

            # Close existing positions if confidence changes
            if self.position:
                if (self.position.is_long and direction_confidence < 0.45) or \
                   (self.position.is_short and direction_confidence > 0.55):
                    self.position.close()
                return  # Skip opening new position this candle

            # Open new positions
            if direction_confidence >= 0.55:  # Strong long signal
                print(f"Opening LONG position: size={size}, price={current_price}")
                sl_price = current_price * (1 - self.stop_loss)
                tp_price = current_price * (1 + self.take_profit)
                self.buy(size=size, sl=sl_price, tp=tp_price)

            elif direction_confidence <= 0.45:  # Strong short signal
                print(f"Opening SHORT position: size={size}, price={current_price}")
                sl_price = current_price * (1 + self.stop_loss)
                tp_price = current_price * (1 - self.take_profit)
                self.sell(size=size, sl=sl_price, tp=tp_price)

        except Exception as e:
            print(f"Error in next: {str(e)}")
            if self.position:  # Close position on error
                self.position.close()

    def _calculate_position_size(self, current_price: float) -> float:
        """Calculate position size based on ATR and risk parameters."""
        try:
            current_idx = len(self.data) - 1

            # Get ATR or use default volatility
            atr = self.atr[current_idx]
            if np.isnan(atr) or atr <= 0:
                atr = current_price * 0.02  # Default to 2% volatility

            # Calculate position size as a fraction of equity (between 0 and 1)
            # Use position_size parameter directly as the fraction
            size = self.position_size  # This is already a fraction (e.g., 0.1 for 10%)

            # Ensure size is within valid range (1% to 25% of equity)
            min_size = 0.01  # Minimum 1% of equity
            max_size = 0.25  # Maximum 25% of equity
            size = max(min_size, min(size, max_size))

            # Adjust size based on volatility
            volatility_factor = atr / current_price
            if volatility_factor > 0.02:  # If volatility is high
                size *= 0.02 / volatility_factor  # Reduce position size

            # Round to 4 decimal places to avoid floating point issues
            size = round(size, 4)

            print(f"Calculated position size: {size} (fraction of equity)")
            return float(size)

        except Exception as e:
            print(f"Error in position sizing: {str(e)}")
            return 0.01  # Return minimum size on error (1% of equity)
