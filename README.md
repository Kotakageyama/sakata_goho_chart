# Trading Bot - 酒田五法 & Transformer Model

暗号通貨取引のための Transformer モデルと酒田五法パターン分析を組み合わせた trading bot プロジェクトです。

## プロジェクト構造

```
sakata_goho_chart/
├── src/                    # 共通のPythonコード
│   ├── __init__.py
│   ├── data_fetcher.py     # データ取得機能
│   ├── technical_indicators.py  # テクニカル指標計算
│   ├── sakata_patterns.py  # 酒田五法パターン検出
│   └── utils.py           # ユーティリティ関数
├── notebooks/
│   ├── training/          # モデルトレーニング用
│   │   └── transformer_model_training.ipynb
│   ├── backtesting/       # バックテスト用
│   │   └── strategy_backtesting.ipynb
│   └── analysis/          # 分析・可視化用
│       └── sakata_pattern_analysis.ipynb
├── models/               # 保存されたモデル
├── data/                 # データファイル
├── config/               # 設定ファイル
│   └── config.yaml
├── results/              # 結果・レポート保存
├── requirements.txt      # 依存関係
└── README.md
```

## セットアップ

1. 依存関係のインストール：

```bash
pip install -r requirements.txt
```

2. API 設定（KuCoin）：

-   KuCoin API キー、シークレット、パスフレーズを取得
-   Google Colab 使用時は userdata に設定
-   ローカル使用時は環境変数または設定ファイルで管理

## 使用方法

### 1. モデルトレーニング

`notebooks/training/transformer_model_training.ipynb`を実行：

-   データ取得
-   テクニカル指標計算
-   Transformer モデル訓練
-   モデル保存

### 2. バックテスト

`notebooks/backtesting/strategy_backtesting.ipynb`を実行：

-   保存されたモデル読み込み
-   戦略バックテスト実行
-   パフォーマンス分析

### 3. 酒田五法分析

`notebooks/analysis/sakata_pattern_analysis.ipynb`を実行：

-   パターン検出
-   有効性分析
-   可視化

## 主要機能

### データ取得 (`src/data_fetcher.py`)

-   取引所 API からの OHLCV データ取得
-   複数通貨ペア対応
-   レート制限対応

### テクニカル指標 (`src/technical_indicators.py`)

-   移動平均線 (SMA)
-   RSI
-   MACD
-   ATR
-   ボリンジャーバンド
-   ta-lib の全指標

### 酒田五法 (`src/sakata_patterns.py`)

-   赤三兵パターン検出
-   黒三兵パターン検出
-   パターン有効性分析
-   チャート可視化

### Transformer モデル

-   マルチヘッドアテンション
-   価格変動方向予測
-   カスタマイズ可能なハイパーパラメータ

### バックテスト機能

-   取引戦略のバックテスト
-   パフォーマンス指標計算
-   Buy & Hold 戦略との比較

## 設定

`config/config.yaml`でプロジェクト設定をカスタマイズ可能：

-   データ設定
-   モデルパラメータ
-   バックテスト設定
-   テクニカル指標設定

## 注意事項

-   実際の取引前に十分なテストを実施してください
-   市場リスクを理解した上で使用してください
-   このプロジェクトは教育目的であり、投資アドバイスではありません

## ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。
