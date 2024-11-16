"""
Improved TransformerStrategy for backtesting with the enhanced model.
Includes dynamic position sizing, risk management, and market regime detection.
"""
from backtesting import Strategy
from backtesting.lib import crossover, cross
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from sklearn.cluster import KMeans

class TransformerStrategy(Strategy):
    """
    Enhanced Transformer-based trading strategy with dynamic risk management
    and market regime detection.
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
        self.volatility = self.I(lambda: self.data.volatility)
        self.volume = self.I(lambda: self.data.Volume)

        # Market regime detection
        self.regime = self.I(self._detect_market_regime)

        # Transaction costs
        self.commission = 0.001  # 0.1% per trade
        self.slippage = 0.0005   # 0.05% slippage

        # Volatility-based position sizing and risk management
        self.volatility_multiplier = 1.5
        self.base_position_size = 0.1
        self.max_position_size = 0.5

        # Dynamic take-profit and stop-loss based on ATR and regime
        self.tp_atr_multiplier = self.I(self._adaptive_tp_multiplier)
        self.sl_atr_multiplier = self.I(self._adaptive_sl_multiplier)

        # Minimum prediction confidence threshold
        self.min_confidence = 0.6

        # Maximum allowed drawdown
        self.max_drawdown = 0.15

        # Portfolio optimization parameters
        self.max_leverage = 2.0
        self.target_volatility = 0.20  # 20% annual volatility target

        # Track portfolio performance
        self.peak_value = self.equity
        self.current_drawdown = 0.0

    def _detect_market_regime(self) -> int:
        """
        Detect market regime using volatility and volume.
        Returns: 0 (low vol), 1 (medium vol), 2 (high vol)
        """
        if len(self.volatility) < 20:
            return 1  # Default to medium volatility regime

        vol = self.volatility[-20:]
        vol_mean = np.mean(vol)
        vol_std = np.std(vol)

        if vol_mean < vol_std:
            return 0  # Low volatility regime
        elif vol_mean > 2 * vol_std:
            return 2  # High volatility regime
        return 1  # Medium volatility regime

    def _adaptive_tp_multiplier(self) -> float:
        """
        Adjust take-profit multiplier based on market regime.
        """
        regime = self.regime[-1]
        base_tp = 3.0
        if regime == 0:  # Low volatility
            return base_tp * 1.5
        elif regime == 2:  # High volatility
            return base_tp * 0.75
        return base_tp

    def _adaptive_sl_multiplier(self) -> float:
        """
        Adjust stop-loss multiplier based on market regime.
        """
        regime = self.regime[-1]
        base_sl = 1.5
        if regime == 0:  # Low volatility
            return base_sl * 0.75
        elif regime == 2:  # High volatility
            return base_sl * 1.5
        return base_sl

    def calculate_position_size(self, confidence: float, current_atr: float) -> float:
        """
        Calculate position size based on prediction confidence, volatility,
        and portfolio optimization constraints.
        """
        # Scale confidence to [0, 1]
        scaled_confidence = max(0, min(1, (confidence - self.min_confidence) / (1 - self.min_confidence)))

        # Adjust for volatility and regime
        regime = self.regime[-1]
        regime_factor = 1.0
        if regime == 0:
            regime_factor = 1.2  # Increase size in low vol
        elif regime == 2:
            regime_factor = 0.8  # Decrease size in high vol

        volatility_factor = 1 / (1 + current_atr * self.volatility_multiplier)

        # Calculate position size with portfolio constraints
        position_size = (
            self.base_position_size *
            scaled_confidence *
            volatility_factor *
            regime_factor
        )

        # Apply leverage and volatility targeting
        target_exposure = self.equity * self.target_volatility / (current_atr * 252**0.5)
        max_size = min(self.max_position_size, target_exposure / self.equity)

        return min(position_size, max_size)

    def calculate_take_profit_stop_loss(self, entry_price: float, current_atr: float) -> Tuple[float, float]:
        """
        Calculate dynamic take-profit and stop-loss levels based on ATR and regime.
        """
        tp_mult = self.tp_atr_multiplier[-1]
        sl_mult = self.sl_atr_multiplier[-1]

        take_profit = entry_price * (1 + current_atr * tp_mult)
        stop_loss = entry_price * (1 - current_atr * sl_mult)
        return take_profit, stop_loss

    def update_drawdown(self):
        """Update drawdown calculations."""
        self.peak_value = max(self.peak_value, self.equity)
        self.current_drawdown = (self.peak_value - self.equity) / self.peak_value

    def should_trade(self, confidence: float, current_price: float) -> bool:
        """
        Determine if trading conditions are met, considering market regime.
        """
        # Check confidence threshold
        if confidence < self.min_confidence:
            return False

        # Check drawdown limit
        self.update_drawdown()
        if self.current_drawdown > self.max_drawdown:
            return False

        # Adjust conditions based on regime
        regime = self.regime[-1]
        if regime == 2:  # High volatility
            if confidence < self.min_confidence * 1.2:  # Require higher confidence
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
        Trading logic implementation with transaction cost consideration
        and regime-based adjustments.
        """
        # Get current predictions and indicators
        price_pred = self.price_predictions[-1]
        direction_conf = self.direction_predictions[-1]
        current_price = self.data.Close[-1]
        current_atr = self.atr[-1]

        # Calculate expected return (accounting for costs)
        transaction_costs = self.commission + self.slippage
        expected_return = (price_pred - current_price) / current_price
        net_expected_return = expected_return - 2 * transaction_costs  # Round trip costs

        # Determine trading direction
        if not self.position:  # No position
            if self.should_trade(direction_conf, current_price):
                # Calculate position size
                size = self.calculate_position_size(direction_conf, current_atr)

                if net_expected_return > 0 and direction_conf > self.min_confidence:
                    # Long entry
                    take_profit, stop_loss = self.calculate_take_profit_stop_loss(current_price, current_atr)
                    self.buy(size=size, sl=stop_loss, tp=take_profit)

                elif net_expected_return < 0 and (1 - direction_conf) > self.min_confidence:
                    # Short entry
                    take_profit, stop_loss = self.calculate_take_profit_stop_loss(current_price, current_atr)
                    self.sell(size=size, sl=stop_loss, tp=take_profit)

        else:  # Position exists
            # Update stop-loss and take-profit based on current ATR and regime
            if self.position.is_long:
                new_tp, new_sl = self.calculate_take_profit_stop_loss(self.position.entry_price, current_atr)
                if new_sl > self.position.sl:  # Trail stop-loss
                    self.position.sl = new_sl
            else:
                new_tp, new_sl = self.calculate_take_profit_stop_loss(self.position.entry_price, current_atr)
                if new_sl < self.position.sl:  # Trail stop-loss for shorts
                    self.position.sl = new_sl

            # Check for position exit based on prediction reversal and regime
            regime = self.regime[-1]
            exit_threshold = 0.3 if regime != 2 else 0.4  # More conservative in high vol

            if self.position.is_long and (expected_return < 0 or direction_conf < exit_threshold):
                self.position.close()
            elif self.position.is_short and (expected_return > 0 or direction_conf > (1 - exit_threshold)):
                self.position.close()
