import pandas as pd
import numpy as np
import pandas_ta as ta
from pyti.bollinger_bands import upper_bollinger_band as bb_up
from pyti.bollinger_bands import middle_bollinger_band as bb_mid
from pyti.bollinger_bands import lower_bollinger_band as bb_low
from ta import add_all_ta_features


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

        # RSI
        df["RSI"] = ta.rsi(df["Close"], length=14)

        # MACD
        df["MACD"] = ta.macd(df["Close"])["MACD_12_26_9"]

        # ATR
        df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)

        return df

    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        ボリンジャーバンドを追加

        Args:
            df: OHLCVデータフレーム
            period: 期間

        Returns:
            pd.DataFrame: ボリンジャーバンドを追加したデータフレーム
        """
        df = df.copy()
        data = df["Close"].values.tolist()

        df["bb_up"] = bb_up(data, period)
        df["bb_mid"] = bb_mid(data, period)
        df["bb_low"] = bb_low(data, period)

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

        # ta-libの全指標を追加
        df = add_all_ta_features(
            df,
            open="Open",
            high="High",
            low="Low",
            close="Close",
            volume="Volume",
            fillna=True,
        )

        # 欠損値の削除
        df.dropna(inplace=True)

        return df

    @staticmethod
    def add_lagged_features(df: pd.DataFrame, lags: list = [1, 2, 3]) -> pd.DataFrame:
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
            df[f"Close{lag}0"] = df["Close"].shift(lag)
            df[f"Open{lag}0"] = df["Open"].shift(lag)

        return df
