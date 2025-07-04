"""
Google Drive マウント & 共通入出力ユーティリティ

Colab での使用方法:
    sys.path.append('/content/drive/MyDrive/kucoin_bot')
    from utils.drive_io import mount_drive, save_data, load_data, DataFetcher
"""

import pandas as pd
import numpy as np
import ccxt
import pickle
import os
import time
from typing import Tuple, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def mount_drive():
    """Google Drive をマウント (Colab専用)"""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("Google Drive がマウントされました")
        return True
    except ImportError:
        print("Google Colab 環境ではありません")
        return False


def get_project_path():
    """プロジェクトパスを取得"""
    if os.path.exists('/content/drive/MyDrive/kucoin_bot'):
        return '/content/drive/MyDrive/kucoin_bot'
    else:
        return '.'


def load_raw(symbol: str = "SOL_USDT", timeframe: str = "1d", limit: int = 1000) -> pd.DataFrame:
    """
    生データを data/raw から読み込み
    
    Args:
        symbol: 通貨ペア (例: 'SOL_USDT')
        timeframe: 時間足 (例: '1d', '1h')
        limit: データ数
        
    Returns:
        pd.DataFrame: OHLCVデータ
    """
    project_path = get_project_path()
    raw_data_dir = os.path.join(project_path, "data", "raw")
    
    # ファイル名の構築
    filename = f"{symbol}_{timeframe}_{limit}.csv"
    filepath = os.path.join(raw_data_dir, filename)
    
    if not os.path.exists(filepath):
        # Parquetファイルも試す
        parquet_filename = f"{symbol}_{timeframe}_{limit}.parquet"
        parquet_filepath = os.path.join(raw_data_dir, parquet_filename)
        
        if os.path.exists(parquet_filepath):
            df = pd.read_parquet(parquet_filepath)
        else:
            raise FileNotFoundError(f"データファイルが見つかりません: {filepath} または {parquet_filepath}")
    else:
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    print(f"✅ 生データを読み込みました: {filepath}")
    print(f"📊 データ形状: {df.shape}")
    print(f"📅 期間: {df.index[0]} ～ {df.index[-1]}")
    
    return df


def save_data(data: Any, filename: str, data_dir: str = "data") -> None:
    """
    データを保存
    
    Args:
        data: 保存するデータ
        filename: ファイル名
        data_dir: データディレクトリ
    """
    project_path = get_project_path()
    save_path = os.path.join(project_path, data_dir)
    os.makedirs(save_path, exist_ok=True)
    
    filepath = os.path.join(save_path, filename)
    
    if filename.endswith('.csv'):
        data.to_csv(filepath, index=True)
    elif filename.endswith('.pkl'):
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    else:
        raise ValueError("サポートされていないファイル形式")
    
    print(f"データを保存しました: {filepath}")


def load_data(filename: str, data_dir: str = "data") -> Any:
    """
    データを読み込み
    
    Args:
        filename: ファイル名
        data_dir: データディレクトリ
        
    Returns:
        読み込んだデータ
    """
    project_path = get_project_path()
    filepath = os.path.join(project_path, data_dir, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"ファイルが見つかりません: {filepath}")
    
    if filename.endswith('.csv'):
        return pd.read_csv(filepath, index_col=0, parse_dates=True)
    elif filename.endswith('.pkl'):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError("サポートされていないファイル形式")


def save_df(df: pd.DataFrame, path: str) -> None:
    """
    DataFrameを保存 (Issue #4 要件対応)
    
    Args:
        df: 保存するDataFrame
        path: 保存パス
    """
    # パスからディレクトリを作成
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    
    if path.endswith('.csv'):
        df.to_csv(path, index=True)
    elif path.endswith('.parquet'):
        df.to_parquet(path, index=True)
    elif path.endswith('.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(df, f)
    else:
        # デフォルトはCSV
        if not path.endswith('.csv'):
            path += '.csv'
        df.to_csv(path, index=True)
    
    print(f"DataFrameを保存しました: {path}")


def load_df(path: str) -> pd.DataFrame:
    """
    DataFrameを読み込み (Issue #4 要件対応)
    
    Args:
        path: 読み込みパス
        
    Returns:
        pd.DataFrame: 読み込んだDataFrame
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"ファイルが見つかりません: {path}")
    
    if path.endswith('.csv'):
        return pd.read_csv(path, index_col=0, parse_dates=True)
    elif path.endswith('.parquet'):
        return pd.read_parquet(path)
    elif path.endswith('.pkl'):
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        # デフォルトはCSVとして読み込み
        return pd.read_csv(path, index_col=0, parse_dates=True)


class DataFetcher:
    """暗号通貨データ取得クラス"""

    def __init__(
        self,
        exchange_name: str = "kucoin",
        api_key: str = None,
        secret: str = None,
        password: str = None,
    ):
        """
        取引所の初期化

        Args:
            exchange_name: 取引所名
            api_key: APIキー
            secret: APIシークレット
            password: APIパスフレーズ
        """
        self.exchange_name = exchange_name
        self.exchange = None

        if exchange_name.lower() == "kucoin":
            self.exchange = ccxt.kucoin(
                {
                    "apiKey": api_key,
                    "secret": secret,
                    "password": password,
                    "enableRateLimit": True,
                }
            )

    def fetch_ohlcv_data(
        self, symbol: str, timeframe: str = "1d", limit: int = 1000
    ) -> pd.DataFrame:
        """
        OHLCVデータを取得

        Args:
            symbol: 通貨ペア (例: 'BTC/USDT')
            timeframe: 時間足 (例: '1d', '1h', '4h')
            limit: 取得するデータ数

        Returns:
            pd.DataFrame: OHLCVデータ
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            data = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            data = data.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                }
            )
            data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms")
            data.set_index("timestamp", inplace=True)

            print(f"データ取得完了: {symbol}, {len(data)}件")
            return data

        except Exception as e:
            print(f"データ取得エラー: {e}")
            return None

    def fetch_multiple_symbols(
        self, symbols: list, timeframe: str = "1d", limit: int = 1000
    ) -> dict:
        """
        複数の通貨ペアのデータを取得

        Args:
            symbols: 通貨ペアのリスト
            timeframe: 時間足
            limit: 取得するデータ数

        Returns:
            dict: {symbol: DataFrame}
        """
        data_dict = {}

        for symbol in symbols:
            data = self.fetch_ohlcv_data(symbol, timeframe, limit)
            if data is not None:
                data_dict[symbol] = data
            time.sleep(1)  # レート制限対策

        return data_dict


class ModelUtils:
    """モデル関連のユーティリティクラス"""

    @staticmethod
    def save_model(model, model_name: str, model_dir: str = "models"):
        """
        モデルを保存

        Args:
            model: 保存するモデル
            model_name: モデル名
            model_dir: モデルディレクトリ
        """
        project_path = get_project_path()
        model_path = os.path.join(project_path, model_dir, model_name)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        if hasattr(model, "save"):
            # Kerasモデルの場合
            model.save(model_path)
        else:
            # scikit-learnモデルの場合
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

        print(f"モデルを保存しました: {model_path}")

    @staticmethod
    def load_model(model_name: str, model_dir: str = "models"):
        """
        モデルを読み込み

        Args:
            model_name: モデル名
            model_dir: モデルディレクトリ

        Returns:
            読み込んだモデル
        """
        project_path = get_project_path()
        model_path = os.path.join(project_path, model_dir, model_name)
        
        if model_name.endswith(".h5") or model_name.endswith(".keras"):
            # Kerasモデルの場合
            from tensorflow.keras.models import load_model
            return load_model(model_path)
        else:
            # scikit-learnモデルの場合
            with open(model_path, "rb") as f:
                return pickle.load(f)


class DataPreprocessor:
    """データ前処理クラス"""

    def __init__(self, scaler_type: str = "standard"):
        """
        初期化

        Args:
            scaler_type: スケーラーの種類 ('standard' or 'minmax')
        """
        self.scaler_type = scaler_type
        self.scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
        self.feature_columns = None

    def fit_transform(self, df: pd.DataFrame, target_column: str = "Close") -> tuple:
        """
        データの正規化とターゲット作成

        Args:
            df: 入力データフレーム
            target_column: ターゲット列名

        Returns:
            tuple: (X_scaled, y)
        """
        # 特徴量列を特定
        self.feature_columns = [col for col in df.columns if col not in [target_column]]

        # 特徴量の抽出
        X = df[self.feature_columns].fillna(0)

        # 正規化
        X_scaled = self.scaler.fit_transform(X)

        # ターゲットの作成（次の日の価格変動方向）
        y = (df[target_column].shift(-1) > df[target_column]).astype(int)

        return X_scaled, y.values

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        学習済みスケーラーでデータを変換

        Args:
            df: 入力データフレーム

        Returns:
            np.ndarray: 正規化されたデータ
        """
        X = df[self.feature_columns].fillna(0)
        return self.scaler.transform(X)

    def save_scaler(self, filename: str = "scaler.pkl"):
        """スケーラーを保存"""
        save_data(self.scaler, filename, "models")

    def load_scaler(self, filename: str = "scaler.pkl"):
        """スケーラーを読み込み"""
        self.scaler = load_data(filename, "models")


class BacktestUtils:
    """バックテスト関連のユーティリティクラス"""

    @staticmethod
    def calculate_returns(
        df: pd.DataFrame, signal_column: str = "signal"
    ) -> pd.DataFrame:
        """
        リターンを計算

        Args:
            df: データフレーム
            signal_column: シグナル列名

        Returns:
            pd.DataFrame: リターンを追加したデータフレーム
        """
        df = df.copy()

        # 日次リターン
        df["returns"] = df["Close"].pct_change()

        # 戦略リターン
        df["strategy_returns"] = df[signal_column].shift(1) * df["returns"]

        # 累積リターン
        df["cumulative_returns"] = (1 + df["returns"]).cumprod()
        df["cumulative_strategy_returns"] = (1 + df["strategy_returns"]).cumprod()

        return df

    @staticmethod
    def calculate_metrics(returns: pd.Series) -> dict:
        """
        パフォーマンス指標を計算

        Args:
            returns: リターンシリーズ

        Returns:
            dict: パフォーマンス指標
        """
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + returns).prod() ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility != 0 else 0
        max_drawdown = (returns.cumsum() - returns.cumsum().cummax()).min()

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
        }