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
        """Initialize strategy parameters and indicators."""
        # Load price data and predictions
        self.close = self.data.Close
        self.price_pred = self.data.price_pred
        self.direction_pred = self.data.direction_pred

        # Technical indicators
        self.sma_fast = self.I(lambda: pd.Series(self.close).rolling(window=10).mean())
        self.sma_slow = self.I(lambda: pd.Series(self.close).rolling(window=20).mean())
        self.rsi = self.I(lambda: self._calculate_rsi_series())
        self.atr = self.I(lambda: self._calculate_atr_series())
        self.volatility = self.I(lambda: pd.Series(self.close).pct_change().rolling(window=20).std())

        # Market regime detection
        self.regime = self.I(self._detect_market_regime)

        # Strategy parameters
        self.min_confidence = 0.6
        self.max_drawdown = 0.15
        self.position_size = 1.0

        # Performance tracking
        self.peak_value = self.equity
        self.current_drawdown = 0.0

        # Risk management
        self.tp_base = 0.03  # Base take profit (3%)
        self.sl_base = 0.02  # Base stop loss (2%)

    def _calculate_rsi_series(self) -> pd.Series:
        """Calculate RSI for the entire series."""
        close_series = pd.Series(self.close)
        delta = close_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        return pd.Series(100 - (100 / (1 + rs)))

    def _calculate_atr_series(self) -> pd.Series:
        """Calculate ATR for the entire series."""
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        close = pd.Series(self.close)
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=14).mean()

    def _detect_market_regime(self) -> np.ndarray:
        """
        Detect market regime using volatility.
        Returns: Array of regime values (0: low vol, 1: medium vol, 2: high vol)
        """
        # Initialize array with medium volatility regime
        regimes = np.ones(len(self.data.Close))

        # Wait for enough data
        if len(self.data.Close) < 20:
            return regimes

        # Calculate rolling volatility statistics
        for i in range(20, len(self.data.Close)):
            # Get volatility window
            vol_window = [v for v in self.volatility[i-20:i] if not pd.isna(v)]

            if len(vol_window) < 2:
                continue

            vol = np.array(vol_window)
            vol_mean = np.mean(vol)
            vol_std = np.std(vol)

            if vol_mean < vol_std:
                regimes[i] = 0  # Low volatility regime
            elif vol_mean > 2 * vol_std:
                regimes[i] = 2  # High volatility regime

        return regimes

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

    def calculate_position_size(self) -> float:
        """
        Calculate position size based on volatility and current regime.
        Returns: Position size as a fraction of portfolio value.
        """
        # Base position size
        base_size = self.position_size

        # Adjust for volatility
        if self.volatility[-1] > 0:
            vol_multiplier = 1 / (self.volatility[-1] * np.sqrt(252))  # Annualized volatility
            base_size *= min(vol_multiplier, 2.0)  # Cap at 2x leverage

        # Adjust for market regime
        regime = self.regime[-1]
        if regime == 0:  # Low volatility
            regime_multiplier = 1.2
        elif regime == 2:  # High volatility
            regime_multiplier = 0.8
        else:  # Medium volatility
            regime_multiplier = 1.0

        # Apply regime adjustment and ensure within limits
        position_size = base_size * regime_multiplier
        return max(min(position_size, 2.0), 0.1)

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
        """Update drawdown tracking."""
        self.peak_value = max(self.peak_value, self.equity)
        if self.peak_value > 0:
            self.current_drawdown = (self.peak_value - self.equity) / self.peak_value

    def should_trade(self, confidence: float, current_price: float) -> bool:
        """
        Determine if trading conditions are met, considering market regime.
        """
        # Check for initialization period
        warmup_period = 20  # Maximum lookback period for indicators
        if len(self.data.Close) < warmup_period:
            return False

        # Check for NaN values in critical indicators
        if (pd.isna(self.sma_fast[-1]) or pd.isna(self.sma_slow[-1]) or
            pd.isna(self.rsi[-1]) or pd.isna(self.atr[-1])):
            return False

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
        Main strategy logic executed on each step.
        """
        # Skip if not enough data for indicators
        if len(self.data) < 20:
            return

        # Get current predictions
        price_pred = self.price_pred[-1]
        direction_pred = self.direction_pred[-1]

        # Calculate prediction confidence
        current_price = self.close[-1]
        price_change = (price_pred - current_price) / current_price
        confidence = abs(price_change)

        # Check if we should trade
        if not self.should_trade(confidence, current_price):
            return

        # Calculate position size and risk levels
        position_size = self.calculate_position_size(confidence, self.atr[-1])
        tp_price, sl_price = self.calculate_take_profit_stop_loss(
            current_price,
            self.atr[-1]
        )

        # Trading decision based on direction prediction
        if direction_pred > 0.5 and not self.position:
            # Long position
            self.buy(size=position_size, sl=sl_price, tp=tp_price)
        elif direction_pred < 0.5 and not self.position:
            # Short position
            self.sell(size=position_size, sl=sl_price, tp=tp_price)

        # Update drawdown tracking
        self.update_drawdown()

        # Close position if maximum drawdown exceeded
        if self.current_drawdown > self.max_drawdown and self.position:
            self.position.close()
