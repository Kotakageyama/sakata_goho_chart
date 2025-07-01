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
# # 02_train_model.ipynb
#
# **目的**: 機械学習モデルの学習とパフォーマンス評価
#
# ## ISSUE #2: モデル学習パイプライン
#
# このノートブックでは以下を行います：
# 1. 前処理済みデータの読み込み
# 2. 特徴量エンジニアリング
# 3. モデル学習（複数アルゴリズム）
# 4. モデル評価・選択
# 5. 学習済みモデルの保存

# %% [markdown]
# ## セットアップ

# %%
# Colab環境でのセットアップ
import sys
import os

try:
    from google.colab import drive
    if '/content/drive' not in [m.mountpoint for m in drive._get_mounts()]:
        drive.mount('/content/drive')
    
    project_path = '/content/drive/MyDrive/kucoin_bot'
    if project_path not in sys.path:
        sys.path.append(project_path)
    
    # 必要なライブラリをインストール
    get_ipython().system('pip install scikit-learn lightgbm xgboost tensorflow')
    
    print("Colab環境でのセットアップ完了")
except ImportError:
    print("ローカル環境で実行中")

# %% [markdown]
# ## ライブラリのインポート

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 機械学習関連
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import lightgbm as lgb
import xgboost as xgb

# ディープラーニング
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlowが利用できません")

# プロジェクト専用ライブラリ
from utils.drive_io import (
    save_data, 
    load_data, 
    ModelUtils,
    DataPreprocessor
)

print("ライブラリのインポート完了")

# %% [markdown]
# ## 設定・パラメータ

# %%
# データファイル
PROCESSED_DATA_FILE = 'processed_crypto_data.pkl'

# モデル設定
TARGET_SYMBOL = 'BTC/USDT'  # 主要対象銘柄
PREDICTION_HORIZON = 1  # 何日後を予測するか
TEST_SIZE = 0.2
RANDOM_STATE = 42

# モデル一覧
MODELS = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    'LightGBM': lgb.LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
    'XGBoost': xgb.XGBClassifier(random_state=RANDOM_STATE),
    'LogisticRegression': LogisticRegression(random_state=RANDOM_STATE),
    'SVM': SVC(random_state=RANDOM_STATE, probability=True)
}

print(f"対象銘柄: {TARGET_SYMBOL}")
print(f"予測期間: {PREDICTION_HORIZON}日後")
print(f"利用可能モデル: {list(MODELS.keys())}")

# %% [markdown]
# ## データの読み込み・準備

# %%
# 処理済みデータの読み込み
print("データを読み込み中...")
processed_data = load_data(PROCESSED_DATA_FILE)

# 対象銘柄のデータを取得
if TARGET_SYMBOL in processed_data:
    df = processed_data[TARGET_SYMBOL].copy()
    print(f"{TARGET_SYMBOL} データ形状: {df.shape}")
else:
    raise ValueError(f"{TARGET_SYMBOL} のデータが見つかりません")

# データの確認
print(f"\nデータ期間: {df.index[0]} - {df.index[-1]}")
print(f"特徴量数: {len(df.columns)}")

# %% [markdown]
# ## ターゲット変数の作成

# %%
# 価格変動方向を予測（上昇=1, 下降=0）
df['target'] = (df['Close'].shift(-PREDICTION_HORIZON) > df['Close']).astype(int)

# 欠損値の処理
df = df.dropna()

print(f"ターゲット変数の分布:")
print(df['target'].value_counts())
print(f"\n正例率: {df['target'].mean():.3f}")

# %% [markdown]
# ## 特徴量とターゲットの分離

# %%
# ターゲット変数以外を特徴量として使用
feature_columns = [col for col in df.columns if col not in ['target']]
X = df[feature_columns]
y = df['target']

print(f"特徴量数: {len(feature_columns)}")
print(f"サンプル数: {len(X)}")

# 特徴量の重要性確認用にカテゴリ分け
basic_features = ['Open', 'High', 'Low', 'Close', 'Volume']
technical_features = [col for col in feature_columns if any(indicator in col for indicator in ['SMA', 'RSI', 'MACD', 'ATR', 'bb_'])]
lag_features = [col for col in feature_columns if 'lag' in col]
pattern_features = [col for col in feature_columns if col in ['doji', 'hammer', 'shooting_star', 'engulfing_bullish', 'engulfing_bearish', 'three_white_soldiers', 'three_black_crows']]

print(f"\n特徴量カテゴリ:")
print(f"  - 基本OHLCV: {len(basic_features)}")
print(f"  - テクニカル指標: {len(technical_features)}")
print(f"  - ラグ特徴量: {len(lag_features)}")
print(f"  - 酒田五法パターン: {len(pattern_features)}")

# %% [markdown]
# ## データの分割・前処理

# %%
# 訓練・テストデータの分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"訓練データ: {X_train.shape}")
print(f"テストデータ: {X_test.shape}")

# データの正規化
preprocessor = DataPreprocessor(scaler_type='standard')
X_train_scaled = preprocessor.scaler.fit_transform(X_train)
X_test_scaled = preprocessor.scaler.transform(X_test)

# スケーラーを保存
preprocessor.save_scaler('model_scaler.pkl')

print("データの前処理完了")

# %% [markdown]
# ## 機械学習モデルの学習・評価

# %%
# モデル評価結果を保存
model_results = {}

for model_name, model in MODELS.items():
    print(f"\n{'='*50}")
    print(f"学習中: {model_name}")
    print(f"{'='*50}")
    
    try:
        # モデル学習
        if model_name in ['RandomForest', 'LightGBM', 'XGBoost']:
            # Tree-based models は正規化不要
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            # 他のモデルは正規化データを使用
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # 評価指標の計算
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # クロスバリデーション
        if model_name in ['RandomForest', 'LightGBM', 'XGBoost']:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        else:
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        
        # 結果を保存
        model_results[model_name] = {
            'model': model,
            'accuracy': accuracy,
            'auc': auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"精度: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"CV精度: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # 分類レポート
        print("\n分類レポート:")
        print(classification_report(y_test, y_pred))
        
    except Exception as e:
        print(f"エラー: {e}")
        model_results[model_name] = {'error': str(e)}

# %% [markdown]
# ## モデル比較・選択

# %%
# 成功したモデルの結果をまとめる
results_df = []
for model_name, results in model_results.items():
    if 'error' not in results:
        results_df.append({
            'Model': model_name,
            'Accuracy': results['accuracy'],
            'AUC': results['auc'],
            'CV_Mean': results['cv_mean'],
            'CV_Std': results['cv_std']
        })

results_df = pd.DataFrame(results_df)
results_df = results_df.sort_values('AUC', ascending=False)

print("モデル比較結果:")
print(results_df.round(4))

# 最良モデルの選択
best_model_name = results_df.iloc[0]['Model']
best_model = model_results[best_model_name]['model']

print(f"\n最良モデル: {best_model_name}")
print(f"AUC: {results_df.iloc[0]['AUC']:.4f}")

# %% [markdown]
# ## 結果の可視化

# %%
# モデル比較のバープロット
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 精度比較
axes[0].bar(results_df['Model'], results_df['Accuracy'])
axes[0].set_title('モデル別精度比較')
axes[0].set_ylabel('Accuracy')
axes[0].tick_params(axis='x', rotation=45)

# AUC比較
axes[1].bar(results_df['Model'], results_df['AUC'])
axes[1].set_title('モデル別AUC比較')
axes[1].set_ylabel('AUC')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 特徴量重要度の分析

# %%
# Tree-based modelの場合は特徴量重要度を表示
if best_model_name in ['RandomForest', 'LightGBM', 'XGBoost']:
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n{best_model_name} 特徴量重要度 (上位20):")
    print(feature_importance.head(20))
    
    # 重要度の可視化
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(20)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('重要度')
    plt.title(f'{best_model_name} 特徴量重要度 (上位20)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## ディープラーニングモデル（オプション）

# %%
if TF_AVAILABLE:
    print("ディープラーニングモデルを学習中...")
    
    # データの形状調整
    X_train_dl = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_dl = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    
    # モデル構築
    model_dl = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_scaled.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        Flatten(),
        Dense(100, activation='relu'),
        Dropout(0.5),
        Dense(50, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model_dl.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # コールバック
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # 学習
    history = model_dl.fit(
        X_train_dl, y_train,
        batch_size=32,
        epochs=100,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # 評価
    y_pred_dl = (model_dl.predict(X_test_dl) > 0.5).astype(int).flatten()
    y_pred_proba_dl = model_dl.predict(X_test_dl).flatten()
    
    dl_accuracy = accuracy_score(y_test, y_pred_dl)
    dl_auc = roc_auc_score(y_test, y_pred_proba_dl)
    
    print(f"\nディープラーニングモデル結果:")
    print(f"精度: {dl_accuracy:.4f}")
    print(f"AUC: {dl_auc:.4f}")
    
    # 結果を追加
    model_results['DeepLearning'] = {
        'model': model_dl,
        'accuracy': dl_accuracy,
        'auc': dl_auc,
        'predictions': y_pred_dl,
        'probabilities': y_pred_proba_dl
    }
else:
    print("TensorFlowが利用できないため、ディープラーニングをスキップ")

# %% [markdown]
# ## モデルの保存

# %%
# 最良モデルを保存
model_utils = ModelUtils()

print("モデルを保存中...")

# 従来の機械学習モデル
model_utils.save_model(best_model, f'best_model_{best_model_name.lower()}.pkl')

# 全モデル結果を保存
save_data(model_results, 'model_results.pkl')

# ディープラーニングモデル（存在する場合）
if TF_AVAILABLE and 'DeepLearning' in model_results:
    model_utils.save_model(model_dl, 'deep_learning_model.h5')

print("モデル保存完了")

# %% [markdown]
# ## まとめ
#
# ### 完了したタスク
# 1. ✅ 前処理済みデータの読み込み
# 2. ✅ ターゲット変数の作成（価格変動方向）
# 3. ✅ 複数モデルでの学習・評価
# 4. ✅ モデル比較・選択
# 5. ✅ 特徴量重要度の分析
# 6. ✅ 学習済みモデルの保存
#
# ### 学習結果
# - 最良モデル: {best_model_name}
# - テスト精度: {model_results[best_model_name]['accuracy']:.4f}
# - AUC: {model_results[best_model_name]['auc']:.4f}
#
# ### 出力ファイル
# - `best_model_{best_model_name.lower()}.pkl`: 最良モデル
# - `model_results.pkl`: 全モデル結果
# - `model_scaler.pkl`: データ正規化スケーラー
# - `deep_learning_model.h5`: ディープラーニングモデル（該当する場合）
#
# ### 次のステップ
# 学習済みモデルは `03_backtest.ipynb` でバックテストに使用されます。