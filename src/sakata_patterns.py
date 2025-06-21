import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf


class SakataPatterns:
    """酒田五法パターン検出クラス"""

    def __init__(self, df: pd.DataFrame):
        """
        初期化

        Args:
            df: OHLCVデータフレーム
        """
        self.df = df.copy()
        self._prepare_data()

    def _prepare_data(self):
        """データの前処理"""
        # ラグ特徴量を追加
        self.df["Close10"] = self.df["Close"].shift(1)
        self.df["Close20"] = self.df["Close"].shift(2)
        self.df["Close30"] = self.df["Close"].shift(3)
        self.df["Open10"] = self.df["Open"].shift(1)
        self.df["Open20"] = self.df["Open"].shift(2)
        self.df["Open30"] = self.df["Open"].shift(3)

        # ボリンジャーバンドがない場合は追加
        if "bb_low" not in self.df.columns:
            from .technical_indicators import TechnicalIndicators

            self.df = TechnicalIndicators.add_bollinger_bands(self.df)

    def detect_aka_sanpei(self) -> pd.DataFrame:
        """
        赤三兵パターンを検出

        Returns:
            pd.DataFrame: 赤三兵パターンが検出された行
        """
        # ルール1: 価格がボリンジャーバンドの下半分にある
        rule_1_mask = self.df["Close"] < ((self.df["bb_low"] + self.df["bb_mid"]) / 2)
        self.df["rule_1"] = rule_1_mask.astype(int)

        # ルール2: 3連続陽線
        rule_2_mask = (
            (self.df["Open"] - self.df["Close"] < 0)
            & (self.df["Open10"] - self.df["Close10"] < 0)
            & (self.df["Open20"] - self.df["Close20"] < 0)
        )
        self.df["rule_2"] = rule_2_mask.astype(int)

        # 両方のルールを満たすレコード
        aka_sanpei_mask = (self.df["rule_1"] == 1) & (self.df["rule_2"] == 1)
        self.df["aka_sanpei"] = aka_sanpei_mask.astype(int)

        return self.df[aka_sanpei_mask]

    def detect_kuro_sanpei(self) -> pd.DataFrame:
        """
        黒三兵パターンを検出

        Returns:
            pd.DataFrame: 黒三兵パターンが検出された行
        """
        # ルール1: 価格がボリンジャーバンドの上半分にある
        rule_1_mask = self.df["Close"] > ((self.df["bb_up"] + self.df["bb_mid"]) / 2)
        self.df["kuro_rule_1"] = rule_1_mask.astype(int)

        # ルール2: 3連続陰線
        rule_2_mask = (
            (self.df["Open"] - self.df["Close"] > 0)
            & (self.df["Open10"] - self.df["Close10"] > 0)
            & (self.df["Open20"] - self.df["Close20"] > 0)
        )
        self.df["kuro_rule_2"] = rule_2_mask.astype(int)

        # 両方のルールを満たすレコード
        kuro_sanpei_mask = (self.df["kuro_rule_1"] == 1) & (self.df["kuro_rule_2"] == 1)
        self.df["kuro_sanpei"] = kuro_sanpei_mask.astype(int)

        return self.df[kuro_sanpei_mask]

    def plot_pattern(self, start_idx: int, end_idx: int, pattern_name: str = ""):
        """
        パターンをチャートで表示

        Args:
            start_idx: 開始インデックス
            end_idx: 終了インデックス
            pattern_name: パターン名
        """
        chart_data = self.df.iloc[start_idx:end_idx]

        # ボリンジャーバンドの追加プロット
        apd = mpf.make_addplot(chart_data[["bb_up", "bb_mid", "bb_low"]])

        # チャート表示
        mpf.plot(
            chart_data[["Open", "High", "Low", "Close", "Volume"]],
            type="candle",
            addplot=apd,
            volume=True,
            style="yahoo",
            title=f"{pattern_name} Pattern Detection",
            figsize=(18, 9),
        )

    def get_pattern_summary(self) -> dict:
        """
        検出されたパターンの要約を取得

        Returns:
            dict: パターンの統計情報
        """
        aka_sanpei_patterns = self.detect_aka_sanpei()
        kuro_sanpei_patterns = self.detect_kuro_sanpei()

        summary = {
            "total_data_points": len(self.df),
            "aka_sanpei_count": len(aka_sanpei_patterns),
            "kuro_sanpei_count": len(kuro_sanpei_patterns),
            "aka_sanpei_percentage": len(aka_sanpei_patterns) / len(self.df) * 100,
            "kuro_sanpei_percentage": len(kuro_sanpei_patterns) / len(self.df) * 100,
        }

        return summary
