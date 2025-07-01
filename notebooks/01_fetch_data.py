# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 01_fetch_data.ipynb
#
# **目的**: 暗号通貨データの取得と前処理
#
# ## ISSUE #1: データ取得・前処理パイプライン
#
# このノートブックでは以下を行います：
# 1. KuCoin APIからOHLCVデータを取得
# 2. テクニカル指標の計算
# 3. 酒田五法パターンの検出
# 4. データのクリーニング・保存

# %% [markdown]
# ## セットアップ
#
# Colab での実行時は以下のコードを実行：

# %%
# Colab環境でのセットアップ
import sys
import os

# Google Drive をマウント（Colab環境のみ）
try:
    from google.colab import drive
    drive.mount('/content/drive')
    
    # プロジェクトパスを追加
    project_path = '/content/drive/MyDrive/kucoin_bot'
    if project_path not in sys.path:
        sys.path.append(project_path)
    
    # 必要なライブラリをインストール
    get_ipython().system('pip install ccxt pandas-ta pyti ta')
    
    print("Colab環境でのセットアップ完了")
except ImportError:
    print("ローカル環境で実行中")
    # ローカル環境用の設定があればここに追加

# %% [markdown]
# ## ライブラリのインポート

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# プロジェクト専用ライブラリ
from utils.drive_io import (
    mount_drive, 
    save_data, 
    load_data, 
    DataFetcher
)
from utils.indicators import (
    TechnicalIndicators, 
    SakataPatterns, 
    create_comprehensive_features
)

# 設定
plt.style.use('default')
sns.set_palette("husl")

# 日本語フォント設定（Colab用）
try:
    get_ipython().system('apt-get -qq -y install fonts-ipafont-gothic')
    import matplotlib.font_manager as fm
    plt.rcParams['font.family'] = 'IPAexGothic'
except:
    pass

print("ライブラリのインポート完了")

# %% [markdown]
# ## 設定・パラメータ

# %%
# API設定（実際の値は環境変数またはファイルから読み込み）
API_CONFIG = {
    'api_key': '',  # KuCoin API Key
    'secret': '',   # KuCoin Secret
    'password': ''  # KuCoin Passphrase
}

# データ取得設定
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'SOL/USDT']
TIMEFRAME = '1d'  # 1日足
LIMIT = 1000  # 取得する足の数

# ファイル名設定
RAW_DATA_FILE = 'raw_crypto_data.pkl'
PROCESSED_DATA_FILE = 'processed_crypto_data.pkl'

print(f"対象銘柄: {SYMBOLS}")
print(f"時間足: {TIMEFRAME}")
print(f"取得期間: {LIMIT}足")

# %% [markdown]
# ## データ取得

# %%
# データフェッチャーの初期化
fetcher = DataFetcher(
    exchange_name='kucoin',
    api_key=API_CONFIG['api_key'],
    secret=API_CONFIG['secret'],
    password=API_CONFIG['password']
)

print("KuCoin接続を初期化しました")

# %%
# 複数銘柄のデータを取得
print("データ取得を開始...")
raw_data = fetcher.fetch_multiple_symbols(
    symbols=SYMBOLS,
    timeframe=TIMEFRAME,
    limit=LIMIT
)

# 取得結果の確認
for symbol, data in raw_data.items():
    if data is not None:
        print(f"{symbol}: {len(data)}件, 期間: {data.index[0]} - {data.index[-1]}")
    else:
        print(f"{symbol}: データ取得失敗")

# 生データを保存
save_data(raw_data, RAW_DATA_FILE)
print(f"\n生データを保存: {RAW_DATA_FILE}")

# %% [markdown]
# ## データの可視化・確認

# %%
# データの基本統計
for symbol, data in raw_data.items():
    if data is not None:
        print(f"\n=== {symbol} ===")
        print(data.describe())

# %%
# 価格チャートの表示
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i, (symbol, data) in enumerate(raw_data.items()):
    if data is not None and i < 4:
        axes[i].plot(data.index, data['Close'], label='Close Price')
        axes[i].set_title(f'{symbol} 価格推移')
        axes[i].set_ylabel('価格 (USDT)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## テクニカル指標の計算

# %%
# 各銘柄にテクニカル指標と酒田五法パターンを追加
processed_data = {}

for symbol, data in raw_data.items():
    if data is not None:
        print(f"\n{symbol} の特徴量を作成中...")
        
        # 包括的な特徴量を作成
        enhanced_data = create_comprehensive_features(data)
        
        print(f"  - 元データ: {len(data)}行, {len(data.columns)}列")
        print(f"  - 処理後: {len(enhanced_data)}行, {len(enhanced_data.columns)}列")
        
        processed_data[symbol] = enhanced_data

print("\n全銘柄の特徴量作成完了")

# %% [markdown]
# ## 処理済みデータの確認

# %%
# 追加された列名を確認
sample_symbol = list(processed_data.keys())[0]
sample_data = processed_data[sample_symbol]

print(f"\n{sample_symbol} の列構成:")
print(f"全列数: {len(sample_data.columns)}")

# 列をカテゴリ別に表示
basic_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
tech_cols = [col for col in sample_data.columns if 'SMA' in col or 'RSI' in col or 'MACD' in col or 'ATR' in col or 'bb_' in col]
lag_cols = [col for col in sample_data.columns if 'lag' in col]
pattern_cols = [col for col in sample_data.columns if col in ['doji', 'hammer', 'shooting_star', 'engulfing_bullish', 'engulfing_bearish', 'three_white_soldiers', 'three_black_crows']]

print(f"\n基本列 ({len(basic_cols)}): {basic_cols}")
print(f"\nテクニカル指標 ({len(tech_cols)}): {tech_cols[:10]}..." if len(tech_cols) > 10 else f"\nテクニカル指標 ({len(tech_cols)}): {tech_cols}")
print(f"\nラグ特徴量 ({len(lag_cols)}): {lag_cols}")
print(f"\n酒田五法パターン ({len(pattern_cols)}): {pattern_cols}")

# %%
# 酒田五法パターンの出現頻度
print("\n酒田五法パターンの出現頻度:")
for symbol, data in processed_data.items():
    print(f"\n=== {symbol} ===")
    for pattern in pattern_cols:
        if pattern in data.columns:
            count = data[pattern].sum()
            percentage = (count / len(data)) * 100
            print(f"  {pattern}: {count}回 ({percentage:.2f}%)")

# %% [markdown]
# ## データの保存

# %%
# 処理済みデータを保存
save_data(processed_data, PROCESSED_DATA_FILE)
print(f"処理済みデータを保存: {PROCESSED_DATA_FILE}")

# 各銘柄を個別ファイルとしても保存
for symbol, data in processed_data.items():
    safe_symbol = symbol.replace('/', '_')
    filename = f"{safe_symbol}_processed.csv"
    save_data(data, filename)
    print(f"  - {symbol} → {filename}")

print("\n全データの保存完了")

# %% [markdown]
# ## データ品質チェック

# %%
# 欠損値のチェック
print("欠損値チェック:")
for symbol, data in processed_data.items():
    missing_count = data.isnull().sum().sum()
    total_values = data.size
    missing_percentage = (missing_count / total_values) * 100
    
    print(f"{symbol}: {missing_count}/{total_values} ({missing_percentage:.2f}%) 欠損")
    
    if missing_count > 0:
        missing_cols = data.columns[data.isnull().any()].tolist()
        print(f"  欠損のある列: {missing_cols[:5]}..." if len(missing_cols) > 5 else f"  欠損のある列: {missing_cols}")

# %% [markdown]
# ## まとめ
#
# ### 完了したタスク
# 1. ✅ KuCoin APIからOHLCVデータを取得
# 2. ✅ テクニカル指標の計算（移動平均、RSI、MACD、ATR、ボリンジャーバンド等）
# 3. ✅ 酒田五法パターンの検出
# 4. ✅ ラグ特徴量の作成
# 5. ✅ データのクリーニング・保存
#
# ### 出力ファイル
# - `raw_crypto_data.pkl`: 生のOHLCVデータ
# - `processed_crypto_data.pkl`: 全特徴量を含む処理済みデータ
# - `{SYMBOL}_processed.csv`: 銘柄別の処理済みデータ
#
# ### 次のステップ
# 処理済みデータは `02_train_model.ipynb` でモデル学習に使用されます。