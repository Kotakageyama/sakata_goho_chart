"""
テクニカル指標のテスト

pytest tests/test_indicators.py で実行
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.indicators import TechnicalIndicators, SakataPatterns


class TestTechnicalIndicators:
    """テクニカル指標のテストクラス"""

    @pytest.fixture
    def sample_data(self):
        """テスト用のサンプルデータを作成"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # 擬似的なOHLCVデータを生成
        close_prices = 100 + np.cumsum(np.random.randn(100) * 0.02)
        
        data = {
            'Open': close_prices + np.random.randn(100) * 0.1,
            'High': close_prices + np.abs(np.random.randn(100) * 0.2),
            'Low': close_prices - np.abs(np.random.randn(100) * 0.2),
            'Close': close_prices,
            'Volume': np.random.randint(1000, 10000, 100)
        }
        
        df = pd.DataFrame(data, index=dates)
        
        # High >= Low >= 0の制約を満たすよう調整
        df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
        df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
        
        return df

    def test_add_basic_indicators(self, sample_data):
        """基本テクニカル指標のテスト"""
        result = TechnicalIndicators.add_basic_indicators(sample_data)
        
        # 期待される列が追加されているかチェック
        expected_columns = ['SMA_10', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'ATR']
        for col in expected_columns:
            assert col in result.columns, f"{col} が追加されていません"
        
        # RSIが0-100の範囲内かチェック
        rsi_values = result['RSI'].dropna()
        assert rsi_values.min() >= 0, "RSIが0未満の値を持っています"
        assert rsi_values.max() <= 100, "RSIが100を超える値を持っています"
        
        # 移動平均線が正の値かチェック
        assert result['SMA_10'].dropna().min() > 0, "SMA_10に負の値があります"
        assert result['SMA_20'].dropna().min() > 0, "SMA_20に負の値があります"

    def test_calculate_rsi(self, sample_data):
        """RSI計算のテスト"""
        rsi = TechnicalIndicators.calculate_rsi(sample_data['Close'])
        
        # RSIが0-100の範囲内かチェック
        rsi_values = rsi.dropna()
        assert len(rsi_values) > 0, "RSI計算結果が空です"
        assert rsi_values.min() >= 0, "RSIが0未満です"
        assert rsi_values.max() <= 100, "RSIが100を超えています"

    def test_add_bollinger_bands(self, sample_data):
        """ボリンジャーバンドのテスト"""
        result = TechnicalIndicators.add_bollinger_bands(sample_data)
        
        # ボリンジャーバンドの列が追加されているかチェック
        bb_columns = ['bb_up', 'bb_mid', 'bb_low']
        for col in bb_columns:
            assert col in result.columns, f"{col} が追加されていません"
        
        # 上限 >= 中央 >= 下限の関係をチェック
        valid_data = result.dropna()
        if len(valid_data) > 0:
            assert (valid_data['bb_up'] >= valid_data['bb_mid']).all(), "上限バンドが中央線より下です"
            assert (valid_data['bb_mid'] >= valid_data['bb_low']).all(), "中央線が下限バンドより下です"

    def test_add_lagged_features(self, sample_data):
        """ラグ特徴量のテスト"""
        result = TechnicalIndicators.add_lagged_features(sample_data, lags=[1, 2])
        
        # ラグ特徴量の列が追加されているかチェック
        expected_lag_columns = ['Close_lag_1', 'Close_lag_2', 'Open_lag_1', 'Open_lag_2']
        for col in expected_lag_columns:
            assert col in result.columns, f"{col} が追加されていません"
        
        # ラグが正しく計算されているかチェック
        assert result['Close_lag_1'].iloc[1] == sample_data['Close'].iloc[0], "ラグ1の計算が正しくありません"


class TestSakataPatterns:
    """酒田五法パターンのテストクラス"""

    @pytest.fixture
    def pattern_data(self):
        """パターン検出用のテストデータ"""
        # 明確なパターンを含むデータを作成
        data = {
            'Open': [100, 101, 99, 102, 98],
            'High': [102, 103, 101, 104, 100],
            'Low': [99, 100, 98, 101, 97],
            'Close': [101, 102, 100, 103, 99],
            'Volume': [1000, 1100, 900, 1200, 800]
        }
        
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        return pd.DataFrame(data, index=dates)

    def test_detect_doji(self, pattern_data):
        """十字線検出のテスト"""
        # 十字線パターンを作成（始値≈終値）
        doji_data = pattern_data.copy()
        doji_data.loc[doji_data.index[2], 'Close'] = doji_data.loc[doji_data.index[2], 'Open']
        
        result = SakataPatterns.detect_doji(doji_data)
        
        assert isinstance(result, pd.Series), "結果がSeries型ではありません"
        assert len(result) == len(doji_data), "結果の長さが元データと異なります"

    def test_detect_hammer(self, pattern_data):
        """ハンマー検出のテスト"""
        result = SakataPatterns.detect_hammer(pattern_data)
        
        assert isinstance(result, pd.Series), "結果がSeries型ではありません"
        assert len(result) == len(pattern_data), "結果の長さが元データと異なります"
        assert result.dtype == bool, "結果がboolean型ではありません"

    def test_detect_engulfing_patterns(self, pattern_data):
        """包み線パターン検出のテスト"""
        bullish_result = SakataPatterns.detect_engulfing_bullish(pattern_data)
        bearish_result = SakataPatterns.detect_engulfing_bearish(pattern_data)
        
        assert isinstance(bullish_result, pd.Series), "強気包み線の結果がSeries型ではありません"
        assert isinstance(bearish_result, pd.Series), "弱気包み線の結果がSeries型ではありません"
        assert len(bullish_result) == len(pattern_data), "強気包み線の結果の長さが異なります"
        assert len(bearish_result) == len(pattern_data), "弱気包み線の結果の長さが異なります"

    def test_add_all_patterns(self, pattern_data):
        """全パターン追加のテスト"""
        result = SakataPatterns.add_all_patterns(pattern_data)
        
        # 期待されるパターン列
        expected_patterns = [
            'doji', 'hammer', 'shooting_star', 
            'engulfing_bullish', 'engulfing_bearish',
            'three_white_soldiers', 'three_black_crows'
        ]
        
        for pattern in expected_patterns:
            assert pattern in result.columns, f"{pattern} が追加されていません"
            assert result[pattern].dtype == bool, f"{pattern} がboolean型ではありません"


class TestIntegration:
    """統合テスト"""

    @pytest.fixture
    def comprehensive_data(self):
        """包括的なテスト用データ"""
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        np.random.seed(42)
        
        close_prices = 100 + np.cumsum(np.random.randn(50) * 0.02)
        
        data = {
            'Open': close_prices + np.random.randn(50) * 0.1,
            'High': close_prices + np.abs(np.random.randn(50) * 0.2),
            'Low': close_prices - np.abs(np.random.randn(50) * 0.2),
            'Close': close_prices,
            'Volume': np.random.randint(1000, 10000, 50)
        }
        
        df = pd.DataFrame(data, index=dates)
        df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
        df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
        
        return df

    def test_create_comprehensive_features(self, comprehensive_data):
        """包括的特徴量作成のテスト"""
        from utils.indicators import create_comprehensive_features
        
        result = create_comprehensive_features(comprehensive_data)
        
        # 元の列数より多くなっているかチェック
        assert len(result.columns) > len(comprehensive_data.columns), "特徴量が追加されていません"
        
        # 基本列が残っているかチェック
        basic_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in basic_columns:
            assert col in result.columns, f"基本列 {col} が失われています"
        
        # データに大きな欠損がないかチェック
        missing_ratio = result.isnull().sum().sum() / result.size
        assert missing_ratio < 0.5, f"欠損値が多すぎます: {missing_ratio:.2%}"


if __name__ == "__main__":
    # テストの実行
    pytest.main([__file__, "-v"])