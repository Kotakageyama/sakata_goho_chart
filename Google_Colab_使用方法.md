# Google Colab での trading bot 使用ガイド

## 🚀 セットアップ手順

### 1. ファイルの準備

1. 生成された `sakata_goho_chart.zip` をダウンロード
2. Google Drive の「マイドライブ」にアップロード

### 2. KuCoin API キーの取得

1. [KuCoin](https://www.kucoin.com/)でアカウント作成
2. API 管理で API キーを作成
3. 必要な権限:
    - General（一般）
    - Spot Trading（現物取引）- 読み取り専用

### 3. Google Colab で API キーを設定

1. Google Colab を開く
2. 左側のパネルの 🔑（キー）アイコンをクリック
3. 以下の 3 つのシークレットを追加:
    - `KuCoin_API_KEY`: API キー
    - `KuCoin_API_SECRET`: API シークレット
    - `KuCoin_API_PASSPHRAS`: API パスフレーズ

### 4. Notebook の実行順序

1. **`google_colab_setup.ipynb`**: 環境セットアップ
2. **`transformer_model_training_colab.ipynb`**: モデル訓練

## 📋 実行手順

### ステップ 1: セットアップ

```python
# 1. Google Colabで google_colab_setup.ipynb を開く
# 2. 「ランタイム」→「すべてのセルを実行」をクリック
# 3. Google Driveのマウント許可を与える
```

### ステップ 2: モデル訓練

```python
# 1. transformer_model_training_colab.ipynb を開く
# 2. 「ランタイム」→「すべてのセルを実行」をクリック
# 3. 訓練完了まで待機（15-30分程度）
```

## 💡 重要なポイント

### GPU 設定

-   「ランタイム」→「ランタイムのタイプを変更」→「GPU」を選択
-   訓練時間を大幅に短縮できます

### ファイル保存場所

訓練完了後、以下のファイルが Google Drive に保存されます：

-   `transformer_model_final.h5`: 最終モデル
-   `transformer_model_checkpoint.h5`: チェックポイント
-   `scaler.pkl`: データスケーラー
-   `training_summary.json`: 訓練結果サマリー
-   `trading_bot_data.csv`: 取得したデータのバックアップ

### 注意事項

1. **API レート制限**: データ取得時に制限に注意
2. **セッション切断**: 長時間実行時のセッション切断に注意
3. **データ保存**: 重要なデータは必ず Google Drive に保存

## 🔧 トラブルシューティング

### よくある問題と解決方法

#### 1. API キーエラー

```
❌ APIキーが設定されていません
```

**解決方法**: Google Colab の Secrets に API キーを正しく設定

#### 2. ZIP ファイルが見つからない

```
ZIPファイルが見つかりません: /content/drive/MyDrive/sakata_goho_chart.zip
```

**解決方法**: ZIP ファイルを Google Drive の正しい場所にアップロード

#### 3. ライブラリインストールエラー

**解決方法**:

-   Google Colab を再起動
-   セルを順番に実行

#### 4. メモリ不足エラー

**解決方法**:

-   データのバッチサイズを小さくする
-   「ランタイム」→「ファクトリーリセット」を実行

## 📊 結果の確認

### 訓練結果の見方

1. **Accuracy**: 予測精度（高いほど良い）
2. **Loss**: 損失（低いほど良い）
3. **Confusion Matrix**: 予測の詳細分析

### 次のステップ

1. モデルが訓練できたら、バックテストを実行
2. 酒田五法分析で伝統的パターンを確認
3. 実際の取引前に十分な検証を実施

## 🎯 成功のコツ

1. **データ品質**: 十分な期間のデータを使用
2. **パラメータ調整**: ハイパーパラメータを調整
3. **検証**: 複数の通貨ペアでテスト
4. **リスク管理**: 必ず適切なリスク管理を実施

## 📞 サポート

問題が発生した場合は、エラーメッセージとともに詳細を報告してください。
