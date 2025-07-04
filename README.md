# KuCoin Bot - 暗号通貨取引ボット

このプロジェクトは、機械学習を用いた暗号通貨取引ボットです。Google Colab での実行を前提とした軽量で効率的な設計になっています。

## 📁 プロジェクト構成

```md
project-root/
├── 01_fetch_data.ipynb          # 📥 KuCoin OHLCVデータ取得 (軽量版)
├── notebooks/
│   ├── 01_fetch_data.py         # ISSUE #1: データ取得・前処理 (包括版)
│   ├── 02_train_model.py        # ISSUE #2: モデル学習
│   └── 03_backtest.py           # ISSUE #3: バックテスト
├── utils/                       # Colab から import する軽量 .py
│   ├── drive_io.py              # GDrive マウント & 共通入出力
│   └── indicators.py            # テクニカル指標など
├── tests/                       # pytest で utils を単体テスト
├── requirements.txt
└── README.md
```

## 🎯 設計思想

- **3つのノートブック**: データ取得 → モデル学習 → バックテストの明確な分離
- **軽量utils**: Colab で `sys.path.append()` 後に簡単import
- **Jupytext管理**: `.py` ファイルでGit管理、Colabでは `.ipynb` 実行
- **実験ブランチ**: 実験セルが肥大化したら新ブランチで複製運用

## 🚀 セットアップ

### Google Colab での実行

1. **プロジェクトをGoogle Driveにアップロード**
   ```bash
   # Google Drive > MyDrive > kucoin_bot/ に配置
   ```

2. **KuCoin API シークレット設定 (重要!)**
   - Google Colab の「シークレット」機能で以下を設定:
     - `KuCoin_API_KEY`: あなたのAPIキー
     - `KuCoin_API_SECRET`: あなたのAPIシークレット  
     - `KuCoin_API_PASSPHRAS`: あなたのAPIパスフレーズ

3. **📥 データ取得 QuickStart (`01_fetch_data.ipynb`)**
   ```python
   # セル1を実行: GDriveマウント & ライブラリインストール
   # セル2を実行: utils/drive_io.py 読み込み
   # セル3を実行: パラメータ設定用widgets表示
   # セル4を実行: データ取得 & 保存実行
   ```
   
   **出力先**: `/content/drive/MyDrive/kucoin_bot/data/raw/`
   - CSV & Parquet 形式で自動保存
   - 取得行数・期間を自動表示
   - データプレビュー表示

4. **従来の詳細セットアップ (包括版)**
   ```python
   # 各ノートブックの最初のセルで実行
   from google.colab import drive
   drive.mount('/content/drive')
   
   import sys
   sys.path.append('/content/drive/MyDrive/kucoin_bot')
   
   # 必要なライブラリをインストール
   !pip install ccxt pandas-ta pyti ta scikit-learn lightgbm xgboost
   ```

5. **ノートブックを順番に実行**
   - `01_fetch_data.ipynb` → `02_train_model.py` → `03_backtest.py`

### ローカル環境での実行

```bash
git clone <repository-url>
cd kucoin_bot
pip install -r requirements.txt

# Jupytextでノートブック変換
pip install jupytext
jupytext --to notebook notebooks/*.py

# テスト実行
pytest tests/
```

## 📊 機能概要

### 01_fetch_data.py
- KuCoin APIからOHLCVデータ取得
- テクニカル指標計算 (SMA, RSI, MACD, ATR, ボリンジャーバンド)
- 酒田五法パターン検出 (十字線、ハンマー、包み線等)
- データクリーニング・保存

### 02_train_model.py
- 複数アルゴリズムでの機械学習 (RandomForest, LightGBM, XGBoost, SVM等)
- ディープラーニングモデル (CNN, LSTM)
- クロスバリデーション・ハイパーパラメータ最適化
- モデル評価・選択・保存

### 03_backtest.py
- 学習済みモデルによる取引シグナル生成
- 包括的バックテスト実行
- パフォーマンス分析 (シャープレシオ、最大ドローダウン等)
- Buy&Hold戦略との比較
- リスク評価・パラメータ最適化

## 🛠 ユーティリティ

### utils/drive_io.py
- Google Drive マウント・パス管理
- データ保存・読み込み (CSV, PKL対応)
- KuCoin APIデータ取得クラス
- モデル・スケーラー保存・読み込み
- バックテスト用ユーティリティ

### utils/indicators.py
- テクニカル指標計算クラス
- 酒田五法パターン検出クラス
- ラグ特徴量生成
- 包括的特徴量作成関数

## 🧪 テスト

```bash
# 単体テスト実行
pytest tests/test_indicators.py -v

# テストカバレッジ確認
pytest tests/ --cov=utils --cov-report=html
```

## 📋 設定例

### API設定 (01_fetch_data.py内)
```python
API_CONFIG = {
    'api_key': 'your-kucoin-api-key',
    'secret': 'your-kucoin-secret', 
    'password': 'your-kucoin-passphrase'
}

SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'SOL/USDT']
TIMEFRAME = '1d'
LIMIT = 1000
```

### モデル設定 (02_train_model.py内)
```python
TARGET_SYMBOL = 'BTC/USDT'
PREDICTION_HORIZON = 1  # 何日後を予測
TEST_SIZE = 0.2
RANDOM_STATE = 42
```

### バックテスト設定 (03_backtest.py内)
```python
INITIAL_CAPITAL = 10000
TRANSACTION_COST = 0.001  # 0.1%
MIN_CONFIDENCE = 0.6
STOP_LOSS = 0.05         # 5%
TAKE_PROFIT = 0.10       # 10%
```

## 📈 出力ファイル

### データファイル
- `raw_crypto_data.pkl`: 生のOHLCVデータ
- `processed_crypto_data.pkl`: 全特徴量付きデータ
- `{SYMBOL}_processed.csv`: 銘柄別処理済みデータ

### モデルファイル
- `best_model_{name}.pkl`: 最良モデル
- `model_results.pkl`: 全モデル評価結果
- `model_scaler.pkl`: データ正規化スケーラー
- `deep_learning_model.h5`: DLモデル (該当時)

### バックテストファイル
- `backtest_results.pkl`: 詳細バックテスト結果
- `trades_history.csv`: 取引履歴

## 🤖 LLM エージェント ワークフロー

### ブランチ運用ルール
**命名規則**: `model/<strategy>/<yyyymmdd>`

```bash
# 例: LSTM戦略の2024年12月1日ブランチ
model/lstm/20241201

# 例: Transformer戦略の2024年12月2日ブランチ  
model/transformer/20241202

# 例: アンサンブル戦略の2024年12月3日ブランチ
model/ensemble/20241203
```

### 自動化ワークフロー

LLMエージェントが以下の手順を自動実行できます:

1. **ブランチ作成** → 2. **学習セル実行** → 3. **PR作成**

```bash
# 🚀 ワンコマンドで完全ワークフロー実行
./scripts/agent_workflow.sh run <strategy>

# 例: LSTM戦略の完全ワークフロー
./scripts/agent_workflow.sh run lstm

# 例: 指定日付でTransformer戦略実行
./scripts/agent_workflow.sh run transformer 20241201
```

### エージェント用コマンド

```bash
# ブランチ作成のみ
./scripts/agent_workflow.sh create <strategy> [date]

# ワークフロー状態確認
./scripts/agent_workflow.sh status

# モデルブランチ一覧
./scripts/agent_workflow.sh list

# ブランチ名検証
./scripts/agent_workflow.sh validate model/lstm/20241201

# ヘルプ表示
./scripts/agent_workflow.sh help
```

### Python API (上級者向け)

```python
# ブランチ管理
from scripts.branch_manager import BranchManager
manager = BranchManager()
branch = manager.create_branch("lstm")

# 完全ワークフロー実行
from scripts.agent_workflow import AgentWorkflow
workflow = AgentWorkflow()
success = workflow.run_workflow("transformer")
```

### 設定ファイル

```bash
# カスタム設定でワークフロー実行
./scripts/agent_workflow.sh run lstm --config config/custom.json

# 設定例をコピー
cp config/agent_workflow.example.json config/my_strategy.json
```

**対応戦略**:
- `lstm`: 時系列予測用LSTM
- `transformer`: Transformer架構
- `ensemble`: アンサンブルモデル
- `technical`: テクニカル分析+ML

## 🔄 従来の開発フロー

1. **ブランチ作成**: 実験用の新ブランチを作成
2. **実験実行**: ノートブックで実験・検証
3. **結果確認**: バックテスト結果を評価
4. **コード整理**: 有効な手法をutilsに統合
5. **テスト追加**: 新機能のテストケース作成
6. **マージ**: メインブランチに統合

## ⚠️ 注意事項

- **投資判断**: このボットの出力は投資助言ではありません
- **リスク管理**: 実運用前に十分な検証とリスク評価を実施
- **API利用**: KuCoin APIの利用規約を遵守
- **データ精度**: 市場データの遅延や精度制限に注意
- **Colab制限**: 実行時間・メモリ制限に留意

## 🤝 コントリビューション

1. Issuesで課題・要望を報告
2. 機能追加は新ブランチで開発
3. テストケースを必ず追加
4. PRでコードレビュー

## 📄 ライセンス

MIT License - 詳細は `LICENSE` ファイルを参照
