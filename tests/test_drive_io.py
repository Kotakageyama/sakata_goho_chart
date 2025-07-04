"""
drive_io.py モジュールのテスト

pytest tests/test_drive_io.py で実行
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.drive_io import (
    mount_drive, save_df, load_df, save_data, load_data,
    get_project_path, DataFetcher, ModelUtils, DataPreprocessor, BacktestUtils
)


class TestDriveIO:
    """drive_io.py の基本機能のテストクラス"""

    @pytest.fixture
    def sample_dataframe(self):
        """テスト用のサンプルDataFrameを作成"""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        np.random.seed(42)
        
        data = {
            'Open': np.random.randn(10) * 0.1 + 100,
            'High': np.random.randn(10) * 0.1 + 101,
            'Low': np.random.randn(10) * 0.1 + 99,
            'Close': np.random.randn(10) * 0.1 + 100,
            'Volume': np.random.randint(1000, 10000, 10)
        }
        
        return pd.DataFrame(data, index=dates)

    @pytest.fixture
    def temp_dir(self):
        """テスト用の一時ディレクトリを作成"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_mount_drive_non_colab(self):
        """Google Colab以外での mount_drive のテスト"""
        # Google Colab環境ではないため、Falseが返されることを確認
        result = mount_drive()
        assert result is False, "非Colab環境でTrueが返されています"

    @patch('utils.drive_io.drive')
    def test_mount_drive_colab_success(self, mock_drive):
        """Google Colab環境での mount_drive 成功のテスト"""
        # google.colabモジュールをモック
        with patch.dict('sys.modules', {'google.colab': MagicMock()}):
            result = mount_drive()
            assert result is True, "Colab環境でFalseが返されています"

    def test_save_df_csv(self, sample_dataframe, temp_dir):
        """CSVファイルでのDataFrame保存のテスト"""
        file_path = os.path.join(temp_dir, 'test_data.csv')
        
        save_df(sample_dataframe, file_path)
        
        assert os.path.exists(file_path), "CSVファイルが保存されていません"
        
        # ファイルサイズをチェック
        assert os.path.getsize(file_path) > 0, "保存されたファイルが空です"

    def test_save_df_parquet(self, sample_dataframe, temp_dir):
        """Parquetファイルでの DataFrame保存のテスト"""
        file_path = os.path.join(temp_dir, 'test_data.parquet')
        
        save_df(sample_dataframe, file_path)
        
        assert os.path.exists(file_path), "Parquetファイルが保存されていません"
        assert os.path.getsize(file_path) > 0, "保存されたファイルが空です"

    def test_save_df_default_extension(self, sample_dataframe, temp_dir):
        """拡張子なしでのDataFrame保存のテスト（デフォルトCSV）"""
        file_path = os.path.join(temp_dir, 'test_data')
        
        save_df(sample_dataframe, file_path)
        
        # デフォルトでCSV拡張子が追加されることを確認
        csv_file_path = file_path + '.csv'
        assert os.path.exists(csv_file_path), "デフォルトCSVファイルが保存されていません"

    def test_load_df_csv(self, sample_dataframe, temp_dir):
        """CSVファイルからのDataFrame読み込みのテスト"""
        file_path = os.path.join(temp_dir, 'test_data.csv')
        
        # まず保存
        save_df(sample_dataframe, file_path)
        
        # 読み込み
        loaded_df = load_df(file_path)
        
        assert isinstance(loaded_df, pd.DataFrame), "読み込み結果がDataFrameではありません"
        assert len(loaded_df) == len(sample_dataframe), "読み込み後のデータ件数が異なります"
        assert list(loaded_df.columns) == list(sample_dataframe.columns), "列名が異なります"

    def test_load_df_parquet(self, sample_dataframe, temp_dir):
        """ParquetファイルからのDataFrame読み込みのテスト"""
        file_path = os.path.join(temp_dir, 'test_data.parquet')
        
        # まず保存
        save_df(sample_dataframe, file_path)
        
        # 読み込み
        loaded_df = load_df(file_path)
        
        assert isinstance(loaded_df, pd.DataFrame), "読み込み結果がDataFrameではありません"
        assert len(loaded_df) == len(sample_dataframe), "読み込み後のデータ件数が異なります"

    def test_load_df_file_not_found(self):
        """存在しないファイルの読み込みテスト"""
        with pytest.raises(FileNotFoundError):
            load_df('non_existent_file.csv')

    def test_get_project_path(self):
        """プロジェクトパス取得のテスト"""
        path = get_project_path()
        assert isinstance(path, str), "プロジェクトパスが文字列ではありません"
        assert len(path) > 0, "プロジェクトパスが空です"


class TestDataFetcher:
    """DataFetcher クラスのテスト"""

    def test_data_fetcher_initialization(self):
        """DataFetcher の初期化テスト"""
        fetcher = DataFetcher()
        assert fetcher.exchange_name == "kucoin", "デフォルト取引所名が正しくありません"

    @patch('ccxt.kucoin')
    def test_data_fetcher_with_credentials(self, mock_kucoin):
        """認証情報付きDataFetcher の初期化テスト"""
        mock_exchange = MagicMock()
        mock_kucoin.return_value = mock_exchange
        
        fetcher = DataFetcher(
            exchange_name="kucoin",
            api_key="test_key",
            secret="test_secret",
            password="test_password"
        )
        
        assert fetcher.exchange is not None, "取引所オブジェクトが初期化されていません"
        mock_kucoin.assert_called_once()


class TestModelUtils:
    """ModelUtils クラスのテスト"""

    @pytest.fixture
    def temp_dir(self):
        """テスト用の一時ディレクトリ"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_save_load_pickle_model(self, temp_dir):
        """pickle形式でのモデル保存・読み込みテスト"""
        # ダミーモデル（辞書）を作成
        dummy_model = {'type': 'test_model', 'parameters': [1, 2, 3]}
        
        model_name = 'test_model.pkl'
        
        # 一時的にget_project_pathをモック
        with patch('utils.drive_io.get_project_path', return_value=temp_dir):
            ModelUtils.save_model(dummy_model, model_name)
            loaded_model = ModelUtils.load_model(model_name)
        
        assert loaded_model == dummy_model, "保存・読み込みでモデルが変わりました"


class TestDataPreprocessor:
    """DataPreprocessor クラスのテスト"""

    @pytest.fixture
    def sample_dataframe(self):
        """テスト用のサンプルDataFrame"""
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        np.random.seed(42)
        
        data = {
            'Open': np.random.randn(20) * 0.1 + 100,
            'High': np.random.randn(20) * 0.1 + 101,
            'Low': np.random.randn(20) * 0.1 + 99,
            'Close': np.random.randn(20) * 0.1 + 100,
            'Volume': np.random.randint(1000, 10000, 20),
            'RSI': np.random.rand(20) * 100
        }
        
        return pd.DataFrame(data, index=dates)

    def test_data_preprocessor_initialization(self):
        """DataPreprocessor の初期化テスト"""
        preprocessor = DataPreprocessor()
        assert preprocessor.scaler_type == "standard", "デフォルトスケーラータイプが正しくありません"
        
        preprocessor_minmax = DataPreprocessor(scaler_type="minmax")
        assert preprocessor_minmax.scaler_type == "minmax", "MinMaxスケーラーが設定されていません"

    def test_fit_transform(self, sample_dataframe):
        """fit_transform のテスト"""
        preprocessor = DataPreprocessor()
        X_scaled, y = preprocessor.fit_transform(sample_dataframe, target_column="Close")
        
        assert X_scaled.shape[0] == len(sample_dataframe), "スケール後のデータ件数が異なります"
        assert len(y) == len(sample_dataframe), "ターゲット配列の長さが異なります"
        assert preprocessor.feature_columns is not None, "特徴量列が設定されていません"


class TestBacktestUtils:
    """BacktestUtils クラスのテスト"""

    @pytest.fixture
    def sample_backtest_data(self):
        """バックテスト用のサンプルデータ"""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        np.random.seed(42)
        
        data = {
            'Close': [100, 101, 99, 102, 98, 103, 97, 104, 96, 105],
            'signal': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        }
        
        return pd.DataFrame(data, index=dates)

    def test_calculate_returns(self, sample_backtest_data):
        """リターン計算のテスト"""
        result = BacktestUtils.calculate_returns(sample_backtest_data)
        
        expected_columns = ['returns', 'strategy_returns', 'cumulative_returns', 'cumulative_strategy_returns']
        for col in expected_columns:
            assert col in result.columns, f"{col} が追加されていません"

    def test_calculate_metrics(self, sample_backtest_data):
        """パフォーマンス指標計算のテスト"""
        df_with_returns = BacktestUtils.calculate_returns(sample_backtest_data)
        returns = df_with_returns['returns'].dropna()
        
        metrics = BacktestUtils.calculate_metrics(returns)
        
        expected_metrics = ['total_return', 'annual_return', 'volatility', 'sharpe_ratio', 'max_drawdown']
        for metric in expected_metrics:
            assert metric in metrics, f"{metric} が計算されていません"
            assert isinstance(metrics[metric], (int, float)), f"{metric} が数値ではありません"


if __name__ == "__main__":
    # テストの実行
    pytest.main([__file__, "-v"])