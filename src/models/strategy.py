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

        return True

    def next(self):
        """Execute trading logic for the current candle."""
        # Update equity peak
        if self.equity > self.equity_peak:
            self.equity_peak = self.equity

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

    def _adaptive_sl_multiplier(self) -> pd.Series:
        """
        Calculate adaptive stop-loss multiplier based on market conditions.
        Returns a series of multipliers.
        """
        # Base multiplier
        base_sl = 1.5

        # Adjust based on volatility
        volatility_factor = pd.Series(self.volatility).fillna(0)
        volatility_adjustment = 1 + volatility_factor

        # Adjust based on RSI
        rsi_series = pd.Series(self.rsi).fillna(50)
        rsi_factor = np.where(rsi_series > 70, 1.2, np.where(rsi_series < 30, 0.8, 1.0))

        return pd.Series(base_sl * volatility_adjustment * rsi_factor)

    def calculate_position_size(self) -> float:
        """
        Calculate position size based on volatility and current regime.
        Returns: Position size as a fraction of portfolio value (0 to 1).
        """
        # Base position size (as a fraction of portfolio)
        base_size = 0.2  # Start with 20% of portfolio

        # Volatility adjustment
        vol = self.volatility[-1]
        if vol > 0:
            # Reduce position size when volatility is high
            vol_multiplier = 1 / (vol * np.sqrt(252))
            base_size *= min(vol_multiplier, 2.0)

        # Regime-based adjustment
        regime = self.regime[-1]
        regime_multiplier = {
            0: 1.2,  # Low volatility regime
            1: 1.0,  # Medium volatility regime
            2: 0.8,  # High volatility regime
        }.get(regime, 1.0)

        # Apply regime multiplier and ensure size is between 0.1 and 1.0
        final_size = min(max(base_size * regime_multiplier, 0.1), 1.0)

        return final_size

    def calculate_take_profit_stop_loss(
        self,
        entry_price: float,
        current_atr: float
    ) -> Tuple[float, float]:
        """
        Calculate adaptive take-profit and stop-loss levels.
        Uses simple percentage-based calculations with ATR scaling.
        Ensures proper price ordering for both long and short positions:
        Long: SL < Entry < TP
        Short: TP < Entry < SL
        """
        # Base percentage moves (1% minimum)
        base_tp_percent = 0.01
        base_sl_percent = 0.01

        # Get current multipliers with minimum values
        tp_mult = max(self.tp_atr_multiplier[-1], 0.5)
        sl_mult = max(self.sl_atr_multiplier[-1], 0.3)

        # Scale percentages by ATR multiplier and current price level
        atr_scale = current_atr / entry_price
        tp_percent = max(base_tp_percent, atr_scale * tp_mult)
        sl_percent = max(base_sl_percent, atr_scale * sl_mult)

        # Calculate prices ensuring proper order relative to entry price
        if self.position.is_long:
            # Long position: SL < Entry < TP
            tp_price = entry_price * (1 + tp_percent)
            sl_price = entry_price * (1 - sl_percent)
        else:
            # Short position: TP < Entry < SL
            # For shorts, we need smaller percentages to maintain proper order
            tp_price = entry_price * (1 - tp_percent * 0.5)  # Closer to entry
            sl_price = entry_price * (1 + sl_percent * 0.5)  # Closer to entry

        # Final validation
        if self.position.is_long:
            assert sl_price < entry_price < tp_price, "Invalid price ordering for long position"
        else:
            assert tp_price < entry_price < sl_price, "Invalid price ordering for short position"

        return tp_price, sl_price

    def update_drawdown(self):
        """Update drawdown tracking."""
        self.peak_value = max(self.peak_value, self.equity)
        if self.peak_value > 0:
            self.current_drawdown = (self.peak_value - self.equity) / self.peak_value

    def should_trade(self) -> bool:
        """
        Determine if trading conditions are met, considering market regime.
        """
        # Check for initialization period
        warmup_period = 20  # Maximum lookback period for indicators
        if len(self.data.Close) < warmup_period:
            return False

        # Check for NaN values in critical indicators
        if (pd.isna(self.rsi[-1]) or pd.isna(self.atr[-1])):
            return False

        # Check drawdown limit
        self.update_drawdown()
        if self.current_drawdown > self.max_drawdown:
            return False

        # Adjust conditions based on regime
        regime = self.regime[-1]
        if regime == 2:  # High volatility
            return False

        return True

    def next(self):
        """
        Main strategy logic executed on each bar.
        """
        # Skip if not enough data
        if len(self.data.Close) < self.n_lookback:
            return

        # Update indicators and market regime
        self._update_indicators()

        # Get current price and ATR
        current_price = self.data.Close[-1]
        current_atr = self.I(self._calculate_atr_series)[-1]

        # Check if we should trade
        if not self.should_trade():
            return

        # Calculate position size
        position_size = self.calculate_position_size()

        # Get predictions from indicators
        price_pred = self.price_pred[-1]
        direction_pred = self.direction_pred[-1]

        # Skip if predictions are not available
        if np.isnan(price_pred) or np.isnan(direction_pred):
            return

        # Calculate take-profit and stop-loss levels
        tp_price, sl_price = self.calculate_take_profit_stop_loss(current_price, current_atr)

        # Trading logic based on predictions
        if not self.position:  # No position
            if direction_pred > 0.6:  # Strong bullish signal
                # Buy at market price
                self.buy(size=position_size, sl=sl_price, tp=tp_price)
            elif direction_pred < 0.4:  # Strong bearish signal
                # Sell at market price
                self.sell(size=position_size, sl=sl_price, tp=tp_price)
        else:
            # Update stop-loss and take-profit for existing position
            if self.position.is_long:
                self.position.sl = min(self.position.sl or float('inf'), sl_price)
                self.position.tp = max(self.position.tp or 0, tp_price)
            else:
                self.position.sl = max(self.position.sl or 0, sl_price)
                self.position.tp = min(self.position.tp or float('inf'), tp_price)

        # Update maximum drawdown
        self.update_drawdown()
