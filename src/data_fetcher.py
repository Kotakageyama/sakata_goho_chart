import pandas as pd
import ccxt
from typing import Tuple, Optional
import time


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
