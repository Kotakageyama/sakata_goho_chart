import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import os


class ModelUtils:
    """モデル関連のユーティリティクラス"""

    @staticmethod
    def save_model(model, model_path: str):
        """
        モデルを保存

        Args:
            model: 保存するモデル
            model_path: 保存パス
        """
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
    def load_model(model_path: str):
        """
        モデルを読み込み

        Args:
            model_path: モデルのパス

        Returns:
            読み込んだモデル
        """
        if model_path.endswith(".h5") or model_path.endswith(".keras"):
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

    def save_scaler(self, path: str):
        """スケーラーを保存"""
        with open(path, "wb") as f:
            pickle.dump(self.scaler, f)
        print(f"スケーラーを保存しました: {path}")

    def load_scaler(self, path: str):
        """スケーラーを読み込み"""
        with open(path, "rb") as f:
            self.scaler = pickle.load(f)
        print(f"スケーラーを読み込みました: {path}")


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
