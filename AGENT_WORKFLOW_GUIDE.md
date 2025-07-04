# 🤖 LLM エージェント ワークフロー ガイド

## Issue #6 実装概要

このドキュメントは、Issue #6「ブランチ運用 & モデル系統管理」の実装を説明し、LLMエージェントが自動化ワークフローを理解・実行できるようにするガイドです。

## 🌱 ブランチ命名規則

### 基本パターン
```
model/<strategy>/<yyyymmdd>
```

### 具体例
- `model/lstm/20241201` - LSTM戦略、2024年12月1日
- `model/transformer/20241202` - Transformer戦略、2024年12月2日
- `model/ensemble/20241203` - アンサンブル戦略、2024年12月3日
- `model/technical/20241204` - テクニカル分析戦略、2024年12月4日

## 🚀 自動化ワークフロー

### 完全自動実行
```bash
# ワンコマンドで「ブランチ作成 → 学習実行 → PR作成」
./scripts/agent_workflow.sh run <strategy>
```

### ステップ別実行
```bash
# 1. ブランチ作成のみ
./scripts/agent_workflow.sh create <strategy>

# 2. 状態確認
./scripts/agent_workflow.sh status

# 3. 完全ワークフロー実行
./scripts/agent_workflow.sh run <strategy>
```

## 📋 対応戦略

| 戦略名 | 説明 | 主要パラメータ |
|--------|------|----------------|
| `lstm` | LSTM時系列予測 | sequence_length, epochs, batch_size |
| `transformer` | Transformer架構 | d_model, num_heads, num_layers |
| `ensemble` | アンサンブルモデル | models, voting |
| `technical` | テクニカル分析+ML | indicators, sakata_patterns |

## 🛠 スクリプト構成

### 1. ブランチ管理 (`scripts/branch_manager.py`)
- ブランチ作成・切り替え・一覧表示
- 命名規則の検証
- 日付自動設定

### 2. ワークフロー実行 (`scripts/agent_workflow.py`)
- 完全自動化ワークフロー
- ノートブック実行
- 自動コミット・PR作成

### 3. シェルラッパー (`scripts/agent_workflow.sh`)
- エージェント向け簡単インターフェース
- 依存関係チェック
- カラー出力

### 4. 設定ファイル (`config/agent_workflow.example.json`)
- 戦略別パラメータ設定
- 実行順序設定
- PR テンプレート

## 💻 LLMエージェント使用例

### パターン1: 新戦略の実験
```bash
# 新しいLSTM戦略実験を開始
./scripts/agent_workflow.sh create lstm

# データ取得から学習まで完全実行
./scripts/agent_workflow.sh run lstm
```

### パターン2: 複数戦略の並行実験
```bash
# 複数戦略を異なる日付で並行実行
./scripts/agent_workflow.sh run lstm 20241201
./scripts/agent_workflow.sh run transformer 20241202  
./scripts/agent_workflow.sh run ensemble 20241203
```

### パターン3: カスタム設定での実行
```bash
# 設定ファイルをコピー・編集
cp config/agent_workflow.example.json config/my_lstm.json

# カスタム設定で実行
./scripts/agent_workflow.sh run lstm --config config/my_lstm.json
```

## 🔍 状態確認・デバッグ

### 現在の状態確認
```bash
./scripts/agent_workflow.sh status
```

### ブランチ一覧
```bash
./scripts/agent_workflow.sh list
```

### ブランチ名検証
```bash
./scripts/agent_workflow.sh validate model/lstm/20241201
```

### ログ確認
```bash
tail -f logs/agent_workflow_$(date +%Y%m%d).log
```

## 📊 ワークフロー実行内容

### 1. 環境検証
- 必要ファイルの存在確認
- Python依存関係チェック
- Git リポジトリ確認

### 2. ブランチ作成・切り替え
- 命名規則に従ったブランチ作成
- 既存ブランチの場合は切り替え

### 3. ノートブック実行
- `01_fetch_data.ipynb` - データ取得
- `02_train_model.ipynb` - モデル学習
- `03_backtest.ipynb` - バックテスト

### 4. 結果管理
- 自動コミット（設定可能）
- PR自動作成（GitHub CLI必要）
- ログファイル生成

## 🔧 カスタマイズ

### 新戦略の追加
1. `config/agent_workflow.example.json` の `strategies` セクションに追加
2. 必要に応じてノートブックを調整
3. 戦略固有のパラメータを設定

### 実行順序の変更
設定ファイルの `execution_order` を編集:
```json
{
  "execution_order": ["fetch_data", "custom_step", "train_model", "backtest"]
}
```

### 通知設定
```json
{
  "notifications": {
    "slack_webhook": "https://hooks.slack.com/...",
    "email": "admin@example.com"
  }
}
```

## ⚠️ 注意事項

### LLMエージェント向け注意点
1. **ブランチ名は必ず規則に従う**: `model/<strategy>/<yyyymmdd>`
2. **同じ日付で複数戦略実行時**: 異なる戦略名を使用
3. **実行前に状態確認**: `./scripts/agent_workflow.sh status`
4. **エラー時はログ確認**: `logs/agent_workflow_*.log`

### 環境要件
- Python 3.7+
- Git
- Jupyter/JupyterLab
- GitHub CLI (PR自動作成時)

### ファイル権限
```bash
# 初回セットアップ時に実行権限付与
chmod +x scripts/agent_workflow.sh
```

## 🎯 期待される効果

1. **一貫性**: 全エージェントが同じブランチ命名規則を使用
2. **追跡性**: 日付ベースでモデル実験を管理
3. **自動化**: 手動操作を最小限に削減
4. **再現性**: 設定ファイルによる実験条件の保存
5. **協調**: 複数エージェントの並行作業支援

## 📚 参考リンク

- [メインREADME](README.md) - プロジェクト全体の説明
- [設定例](config/agent_workflow.example.json) - ワークフロー設定
- [GitHub Issues](https://github.com/[repo]/issues/6) - 元のIssue #6

---
*このガイドは Issue #6 の実装として作成されました。*
*LLMエージェントが理解しやすいよう、具体例と明確な手順を重視しています。*