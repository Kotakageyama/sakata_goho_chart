import nbformat as nbf

# Create a new notebook
nb = nbf.v4.new_notebook()

# Create cells
cells = []

# Title and description
cells.append(nbf.v4.new_markdown_cell('''# 暗号通貨価格予測モデル (Transformer)

このノートブックでは、改善されたTransformerモデルを使用して暗号通貨の価格予測を行います。
Google DriveとローカルCSVの両方からデータを読み込むことができます。'''))

# Install dependencies
cells.append(nbf.v4.new_code_cell('''# 必要なライブラリのインストール
!pip install tensorflow pandas numpy scikit-learn backtesting ta'''))

# Mount Google Drive
cells.append(nbf.v4.new_code_cell('''# Google Driveのマウント
from google.colab import drive
drive.mount("/content/drive")'''))

# Clone repository
cells.append(nbf.v4.new_code_cell('''# リポジトリのクローン
!git clone https://github.com/Kotakageyama/sakata_goho_chart.git
%cd sakata_goho_chart'''))

# Import modules
cells.append(nbf.v4.new_code_cell('''# 必要なモジュールのインポート
from src.data import CryptoDataLoader
from src.models.transformer_model import create_model
from src.models.training import ModelTrainer
from src.models.strategy import TransformerStrategy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline'''))

# Data loading section
cells.append(nbf.v4.new_markdown_cell('''## データの読み込み
Google DriveまたはローカルCSVからデータを読み込むことができます。'''))

cells.append(nbf.v4.new_code_cell('''# データローダーの初期化
data_loader = CryptoDataLoader()

# Google Driveからデータを読み込む場合
drive_path = "/content/drive/MyDrive/Colab Notebooks/DataSetBox/ETHUSD_2Year_2022-11-15_2024-11-15.csv"
df_drive = data_loader.load_data(drive_path)
print("Google Driveからのデータ形状:", df_drive.shape)
print("\\nGoogle Driveデータのサンプル:")
display(df_drive.head())

# ローカルCSVからデータを読み込む場合
local_path = "data/ETHUSD_2Year_2022-11-15_2024-11-15.csv"
df_local = data_loader.load_data(local_path)
print("\\nローカルCSVからのデータ形状:", df_local.shape)
print("\\nローカルデータのサンプル:")
display(df_local.head())'''))

# Model training section
cells.append(nbf.v4.new_markdown_cell('## モデルのトレーニング'))

cells.append(nbf.v4.new_code_cell('''# データの準備
sequence_length = 60
X_train, X_test, y_train, y_test, scaler = data_loader.prepare_data(
    df_drive,
    sequence_length=sequence_length
)

# モデルトレーナーの初期化
trainer = ModelTrainer(
    sequence_length=sequence_length,
    num_features=X_train.shape[2],
    n_splits=5
)

# ハイパーパラメータの最適化
param_grid = {
    "d_model": [64, 128],
    "num_heads": [4, 8],
    "num_transformer_blocks": [2, 4],
    "dropout": [0.1, 0.2]
}

best_params, best_metrics = trainer.optimize_hyperparameters(
    X_train,
    y_train,
    param_grid,
    epochs=50,
    batch_size=32,
    validation_split=0.2
)

print("最適なパラメータ:", best_params)
print("最良のメトリクス:", best_metrics)'''))

# Backtesting section
cells.append(nbf.v4.new_markdown_cell('## バックテストの実行'))

cells.append(nbf.v4.new_code_cell('''from backtesting import Backtest

# モデルの予測を取得
predictions = trainer.best_model.predict(X_test)
price_pred = predictions[0]
direction_pred = predictions[1]

# バックテストデータの準備
test_data = df_drive.iloc[-len(y_test):].copy()
test_data["price_pred"] = price_pred
test_data["direction_pred"] = direction_pred

# バックテストの実行
bt = Backtest(
    test_data,
    TransformerStrategy,
    cash=10000,
    commission=.002,
    trade_on_close=True,
)

# 結果の表示
stats = bt.run()
print(stats)
bt.plot()'''))

# Add cells to notebook
nb['cells'] = cells

# Write notebook to file
nbf.write(nb, 'CryptoBOT_Transformer_Colab.ipynb')
