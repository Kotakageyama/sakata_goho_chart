"""
テクニカル指標 & 酒田五法パターン検出

Colab での使用方法:
    sys.path.append('/content/drive/MyDrive/kucoin_bot')
    from utils.indicators import TechnicalIndicators, SakataPatterns
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional

try:
    import pandas_ta as ta
    from pyti.bollinger_bands import upper_bollinger_band as bb_up
    from pyti.bollinger_bands import middle_bollinger_band as bb_mid
    from pyti.bollinger_bands import lower_bollinger_band as bb_low
    from ta import add_all_ta_features
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    print("警告: ta-lib, pandas_ta, pyti がインストールされていません")


class TechnicalIndicators:
    """テクニカル指標計算クラス"""

    @staticmethod
    def add_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        基本的なテクニカル指標を追加

        Args:
            df: OHLCVデータフレーム

        Returns:
            pd.DataFrame: テクニカル指標を追加したデータフレーム
        """
        df = df.copy()

        # 移動平均線
        df["SMA_10"] = df["Close"].rolling(window=10).mean()
        df["SMA_20"] = df["Close"].rolling(window=20).mean()
        df["SMA_50"] = df["Close"].rolling(window=50).mean()

        # 自前RSI計算
        df["RSI"] = TechnicalIndicators.calculate_rsi(df["Close"])

        # 自前MACD計算
        macd_data = TechnicalIndicators.calculate_macd(df["Close"])
        df["MACD"] = macd_data

        # ATR（Average True Range）
        df["ATR"] = TechnicalIndicators.calculate_atr(df["High"], df["Low"], df["Close"])

        return df

    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """RSIを計算"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """MACDを計算"""
        ema_fast = series.ewm(span=fast).mean()
        ema_slow = series.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd

    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """ATRを計算"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: int = 2) -> pd.DataFrame:
        """
        ボリンジャーバンドを追加

        Args:
            df: OHLCVデータフレーム
            period: 期間
            std_dev: 標準偏差の倍数

        Returns:
            pd.DataFrame: ボリンジャーバンドを追加したデータフレーム
        """
        df = df.copy()
        
        # 自前でボリンジャーバンドを計算
        sma = df["Close"].rolling(window=period).mean()
        std = df["Close"].rolling(window=period).std()
        
        df["bb_up"] = sma + (std * std_dev)
        df["bb_mid"] = sma
        df["bb_low"] = sma - (std * std_dev)

        return df

    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        全てのテクニカル指標を追加

        Args:
            df: OHLCVデータフレーム

        Returns:
            pd.DataFrame: 全テクニカル指標を追加したデータフレーム
        """
        df = df.copy()

        # 基本指標を追加
        df = TechnicalIndicators.add_basic_indicators(df)

        # ボリンジャーバンドを追加
        df = TechnicalIndicators.add_bollinger_bands(df)

        # ta-libが利用可能な場合は追加指標を計算
        if TA_AVAILABLE:
            try:
                df = add_all_ta_features(
                    df,
                    open="Open",
                    high="High",
                    low="Low",
                    close="Close",
                    volume="Volume",
                    fillna=True,
                )
            except Exception as e:
                print(f"ta-lib指標の追加でエラー: {e}")

        # 欠損値の削除
        df.dropna(inplace=True)

        return df

    @staticmethod
    def add_lagged_features(df: pd.DataFrame, lags: List[int] = [1, 2, 3]) -> pd.DataFrame:
        """
        ラグ特徴量を追加

        Args:
            df: データフレーム
            lags: ラグ数のリスト

        Returns:
            pd.DataFrame: ラグ特徴量を追加したデータフレーム
        """
        df = df.copy()

        for lag in lags:
            df[f"Close_lag_{lag}"] = df["Close"].shift(lag)
            df[f"Open_lag_{lag}"] = df["Open"].shift(lag)

        return df


class SakataPatterns:
    """酒田五法パターン検出クラス"""

    @staticmethod
    def detect_doji(df: pd.DataFrame, threshold: float = 0.1) -> pd.Series:
        """
        十字線（同事線）を検出

        Args:
            df: OHLCVデータフレーム
            threshold: 実体の閾値（%）

        Returns:
            pd.Series: 十字線フラグ
        """
        body_ratio = abs(df["Close"] - df["Open"]) / (df["High"] - df["Low"])
        return body_ratio < threshold

    @staticmethod
    def detect_hammer(df: pd.DataFrame) -> pd.Series:
        """
        ハンマー（カラカサ）を検出

        Returns:
            pd.Series: ハンマーフラグ
        """
        body = abs(df["Close"] - df["Open"])
        high_shadow = df["High"] - df[["Close", "Open"]].max(axis=1)
        low_shadow = df[["Close", "Open"]].min(axis=1) - df["Low"]

        # 実体が小さく、下ヒゲが長い
        small_body = body < (df["High"] - df["Low"]) * 0.3
        long_lower_shadow = low_shadow > body * 2
        short_upper_shadow = high_shadow < body * 0.5

        return small_body & long_lower_shadow & short_upper_shadow

    @staticmethod
    def detect_shooting_star(df: pd.DataFrame) -> pd.Series:
        """
        流れ星を検出

        Returns:
            pd.Series: 流れ星フラグ
        """
        body = abs(df["Close"] - df["Open"])
        high_shadow = df["High"] - df[["Close", "Open"]].max(axis=1)
        low_shadow = df[["Close", "Open"]].min(axis=1) - df["Low"]

        # 実体が小さく、上ヒゲが長い
        small_body = body < (df["High"] - df["Low"]) * 0.3
        long_upper_shadow = high_shadow > body * 2
        short_lower_shadow = low_shadow < body * 0.5

        return small_body & long_upper_shadow & short_lower_shadow

    @staticmethod
    def detect_engulfing_bullish(df: pd.DataFrame) -> pd.Series:
        """
        強気包み線を検出

        Returns:
            pd.Series: 強気包み線フラグ
        """
        # 前日が陰線、当日が陽線
        prev_bearish = df["Close"].shift(1) < df["Open"].shift(1)
        curr_bullish = df["Close"] > df["Open"]

        # 当日の実体が前日の実体を包む
        curr_open_lower = df["Open"] < df["Close"].shift(1)
        curr_close_higher = df["Close"] > df["Open"].shift(1)

        return prev_bearish & curr_bullish & curr_open_lower & curr_close_higher

    @staticmethod
    def detect_engulfing_bearish(df: pd.DataFrame) -> pd.Series:
        """
        弱気包み線を検出

        Returns:
            pd.Series: 弱気包み線フラグ
        """
        # 前日が陽線、当日が陰線
        prev_bullish = df["Close"].shift(1) > df["Open"].shift(1)
        curr_bearish = df["Close"] < df["Open"]

        # 当日の実体が前日の実体を包む
        curr_open_higher = df["Open"] > df["Close"].shift(1)
        curr_close_lower = df["Close"] < df["Open"].shift(1)

        return prev_bullish & curr_bearish & curr_open_higher & curr_close_lower

    @staticmethod
    def detect_three_white_soldiers(df: pd.DataFrame) -> pd.Series:
        """
        三兵（赤三兵）を検出

        Returns:
            pd.Series: 三兵フラグ
        """
        # 3日連続陽線
        bullish_1 = df["Close"] > df["Open"]
        bullish_2 = df["Close"].shift(1) > df["Open"].shift(1)
        bullish_3 = df["Close"].shift(2) > df["Open"].shift(2)

        # 各日の終値が前日の終値より高い
        higher_close_1 = df["Close"] > df["Close"].shift(1)
        higher_close_2 = df["Close"].shift(1) > df["Close"].shift(2)

        return bullish_1 & bullish_2 & bullish_3 & higher_close_1 & higher_close_2

    @staticmethod
    def detect_three_black_crows(df: pd.DataFrame) -> pd.Series:
        """
        三羽烏（黒三兵）を検出

        Returns:
            pd.Series: 三羽烏フラグ
        """
        # 3日連続陰線
        bearish_1 = df["Close"] < df["Open"]
        bearish_2 = df["Close"].shift(1) < df["Open"].shift(1)
        bearish_3 = df["Close"].shift(2) < df["Open"].shift(2)

        # 各日の終値が前日の終値より低い
        lower_close_1 = df["Close"] < df["Close"].shift(1)
        lower_close_2 = df["Close"].shift(1) < df["Close"].shift(2)

        return bearish_1 & bearish_2 & bearish_3 & lower_close_1 & lower_close_2

    @staticmethod
    def add_all_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """
        全ての酒田五法パターンを検出

        Args:
            df: OHLCVデータフレーム

        Returns:
            pd.DataFrame: パターンフラグを追加したデータフレーム
        """
        df = df.copy()

        df["doji"] = SakataPatterns.detect_doji(df)
        df["hammer"] = SakataPatterns.detect_hammer(df)
        df["shooting_star"] = SakataPatterns.detect_shooting_star(df)
        df["engulfing_bullish"] = SakataPatterns.detect_engulfing_bullish(df)
        df["engulfing_bearish"] = SakataPatterns.detect_engulfing_bearish(df)
        df["three_white_soldiers"] = SakataPatterns.detect_three_white_soldiers(df)
        df["three_black_crows"] = SakataPatterns.detect_three_black_crows(df)

        return df


def create_comprehensive_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    包括的な特徴量を作成
    
    Args:
        df: OHLCVデータフレーム
        
    Returns:
        pd.DataFrame: 全特徴量を追加したデータフレーム
    """
    df = df.copy()
    
    # テクニカル指標を追加
    df = TechnicalIndicators.add_all_indicators(df)
    
    # ラグ特徴量を追加
    df = TechnicalIndicators.add_lagged_features(df)
    
    # 酒田五法パターンを追加
    df = SakataPatterns.add_all_patterns(df)
    
    return df

def add_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    全ての特徴量を追加する統一インターフェース
    
    Args:
        df: OHLCVデータフレーム
        
    Returns:
        pd.DataFrame: 全特徴量を追加したデータフレーム
    """
    return create_comprehensive_features(df)