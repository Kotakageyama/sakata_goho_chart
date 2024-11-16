"""
Improved TransformerStrategy for backtesting with the enhanced model.
Includes dynamic position sizing and risk management.
"""
from backtesting import Strategy
from backtesting.lib import crossover, cross
import numpy as np
import pandas as pd
from typing import Optional, Tuple

class TransformerStrategy(Strategy):
    """
    Enhanced Transformer-based trading strategy with dynamic risk management.
    """
    def init(self):
        # Price predictions and direction signals
        self.price_predictions = self.I(lambda: self.data.price_pred)
        self.direction_predictions = self.I(lambda: self.data.direction_pred)

        # Technical indicators for confirmation
        self.sma_fast = self.I(lambda: self.data.SMA_10)
        self.sma_slow = self.I(lambda: self.data.SMA_20)
        self.rsi = self.I(lambda: self.data.RSI)
        self.atr = self.I(lambda: self.data.ATR)

        # Volatility-based position sizing and risk management
        self.volatility_multiplier = 1.5
        self.base_position_size = 0.1
        self.max_position_size = 0.5

        # Dynamic take-profit and stop-loss based on ATR
        self.tp_atr_multiplier = 3.0
        self.sl_atr_multiplier = 1.5

        # Minimum prediction confidence threshold
        self.min_confidence = 0.6

        # Maximum allowed drawdown
        self.max_drawdown = 0.15

        # Track portfolio performance
        self.peak_value = self.equity
        self.current_drawdown = 0.0

    def calculate_position_size(self, confidence: float, current_atr: float) -> float:
        """
        Calculate position size based on prediction confidence and volatility.
        """
        # Scale confidence to [0, 1]
        scaled_confidence = max(0, min(1, (confidence - self.min_confidence) / (1 - self.min_confidence)))

        # Adjust for volatility
        volatility_factor = 1 / (1 + current_atr * self.volatility_multiplier)

        # Calculate final position size
        position_size = self.base_position_size * scaled_confidence * volatility_factor
        return min(position_size, self.max_position_size)

    def calculate_take_profit_stop_loss(self, entry_price: float, current_atr: float) -> Tuple[float, float]:
        """
        Calculate dynamic take-profit and stop-loss levels based on ATR.
        """
        take_profit = entry_price * (1 + current_atr * self.tp_atr_multiplier)
        stop_loss = entry_price * (1 - current_atr * self.sl_atr_multiplier)
        return take_profit, stop_loss

    def update_drawdown(self):
        """
        Update drawdown calculations.
        """
        self.peak_value = max(self.peak_value, self.equity)
        self.current_drawdown = (self.peak_value - self.equity) / self.peak_value

    def should_trade(self, confidence: float, current_price: float) -> bool:
        """
        Determine if trading conditions are met.
        """
        # Check confidence threshold
        if confidence < self.min_confidence:
            return False

        # Check drawdown limit
        self.update_drawdown()
        if self.current_drawdown > self.max_drawdown:
            return False

        # Check technical confirmations
        trend_aligned = (
            self.sma_fast[-1] > self.sma_slow[-1] if confidence > 0.5
            else self.sma_fast[-1] < self.sma_slow[-1]
        )

        # Check RSI extremes
        rsi_ok = 20 < self.rsi[-1] < 80

        return trend_aligned and rsi_ok

    def next(self):
        """
        Trading logic implementation.
        """
        # Get current predictions and indicators
        price_pred = self.price_predictions[-1]
        direction_conf = self.direction_predictions[-1]
        current_price = self.data.Close[-1]
        current_atr = self.atr[-1]

        # Calculate expected return
        expected_return = (price_pred - current_price) / current_price

        # Determine trading direction
        if not self.position:  # No position
            if self.should_trade(direction_conf, current_price):
                # Calculate position size
                size = self.calculate_position_size(direction_conf, current_atr)

                if expected_return > 0 and direction_conf > self.min_confidence:
                    # Long entry
                    take_profit, stop_loss = self.calculate_take_profit_stop_loss(current_price, current_atr)
                    self.buy(size=size, sl=stop_loss, tp=take_profit)


                elif expected_return < 0 and (1 - direction_conf) > self.min_confidence:
                    # Short entry
                    take_profit, stop_loss = self.calculate_take_profit_stop_loss(current_price, current_atr)
                    self.sell(size=size, sl=stop_loss, tp=take_profit)

        else:  # Position exists
            # Update stop-loss and take-profit based on current ATR
            if self.position.is_long:
                new_tp, new_sl = self.calculate_take_profit_stop_loss(self.position.entry_price, current_atr)
                if new_sl > self.position.sl:  # Trail stop-loss
                    self.position.sl = new_sl
            else:
                new_tp, new_sl = self.calculate_take_profit_stop_loss(self.position.entry_price, current_atr)
                if new_sl < self.position.sl:  # Trail stop-loss for shorts
                    self.position.sl = new_sl

            # Check for position exit based on prediction reversal
            if self.position.is_long and expected_return < 0 and direction_conf < 0.3:
                self.position.close()
            elif self.position.is_short and expected_return > 0 and direction_conf > 0.7:
                self.position.close()
