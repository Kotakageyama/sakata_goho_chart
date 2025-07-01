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
# # 03_backtest.ipynb
#
# **目的**: 学習済みモデルを使用した取引戦略のバックテスト
#
# ## ISSUE #3: バックテスト・戦略評価パイプライン
#
# このノートブックでは以下を行います：
# 1. 学習済みモデルの読み込み
# 2. 取引シグナルの生成
# 3. バックテストの実行
# 4. パフォーマンス分析
# 5. リスク評価・最適化

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
    get_ipython().system('pip install plotly quantlib-python pyfolio')
    
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

# バックテスト関連
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotlyが利用できません")

# プロジェクト専用ライブラリ
from utils.drive_io import (
    save_data, 
    load_data, 
    ModelUtils,
    DataPreprocessor,
    BacktestUtils
)

print("ライブラリのインポート完了")

# %% [markdown]
# ## 設定・パラメータ

# %%
# データ・モデルファイル
PROCESSED_DATA_FILE = 'processed_crypto_data.pkl'
MODEL_RESULTS_FILE = 'model_results.pkl'
SCALER_FILE = 'model_scaler.pkl'

# バックテスト設定
TARGET_SYMBOL = 'BTC/USDT'
INITIAL_CAPITAL = 10000  # 初期資金（USDT）
TRANSACTION_COST = 0.001  # 取引手数料（0.1%）
MIN_CONFIDENCE = 0.6  # 最小信頼度閾値

# リスク管理
STOP_LOSS = 0.05  # ストップロス（5%）
TAKE_PROFIT = 0.10  # テイクプロフィット（10%）
MAX_POSITION_SIZE = 0.2  # 最大ポジションサイズ（20%）

print(f"対象銘柄: {TARGET_SYMBOL}")
print(f"初期資金: ${INITIAL_CAPITAL:,.2f}")
print(f"取引手数料: {TRANSACTION_COST*100:.2f}%")

# %% [markdown]
# ## データ・モデルの読み込み

# %%
# データの読み込み
print("データを読み込み中...")
processed_data = load_data(PROCESSED_DATA_FILE)
model_results = load_data(MODEL_RESULTS_FILE)

# 対象銘柄のデータを取得
if TARGET_SYMBOL in processed_data:
    df = processed_data[TARGET_SYMBOL].copy()
    print(f"{TARGET_SYMBOL} データ形状: {df.shape}")
    print(f"データ期間: {df.index[0]} - {df.index[-1]}")
else:
    raise ValueError(f"{TARGET_SYMBOL} のデータが見つかりません")

# スケーラーの読み込み
preprocessor = DataPreprocessor()
preprocessor.load_scaler(SCALER_FILE)

print("データ読み込み完了")

# %%
# 最良モデルの特定と読み込み
best_model_info = None
best_auc = 0

for model_name, results in model_results.items():
    if 'error' not in results and 'auc' in results:
        if results['auc'] > best_auc:
            best_auc = results['auc']
            best_model_info = (model_name, results)

if best_model_info is None:
    raise ValueError("利用可能なモデルが見つかりません")

best_model_name, best_results = best_model_info
best_model = best_results['model']

print(f"最良モデル: {best_model_name}")
print(f"AUC: {best_auc:.4f}")

# %% [markdown]
# ## 取引シグナルの生成

# %%
# 特徴量の準備（ターゲット変数を除く）
feature_columns = [col for col in df.columns if col not in ['target']]
X = df[feature_columns].fillna(0)

# データの正規化（必要な場合）
if best_model_name not in ['RandomForest', 'LightGBM', 'XGBoost']:
    X_scaled = preprocessor.transform(X)
else:
    X_scaled = X.values

print(f"特徴量数: {len(feature_columns)}")
print(f"予測対象期間: {len(X)}日")

# %%
# 予測の実行
print("取引シグナルを生成中...")

# 予測確率を取得
if hasattr(best_model, 'predict_proba'):
    if best_model_name not in ['RandomForest', 'LightGBM', 'XGBoost']:
        prediction_probs = best_model.predict_proba(X_scaled)[:, 1]
    else:
        prediction_probs = best_model.predict_proba(X)[:, 1]
else:
    # 確率が取得できない場合は予測値を使用
    if best_model_name not in ['RandomForest', 'LightGBM', 'XGBoost']:
        predictions = best_model.predict(X_scaled)
    else:
        predictions = best_model.predict(X)
    prediction_probs = predictions.astype(float)

# シグナルの生成
df['prediction_prob'] = prediction_probs
df['signal'] = 0  # デフォルトは無ポジション

# シグナルロジック
buy_condition = (df['prediction_prob'] > MIN_CONFIDENCE)
sell_condition = (df['prediction_prob'] < (1 - MIN_CONFIDENCE))

df.loc[buy_condition, 'signal'] = 1   # 買いシグナル
df.loc[sell_condition, 'signal'] = -1  # 売りシグナル

# シグナル統計
signal_counts = df['signal'].value_counts()
print(f"\nシグナル統計:")
print(f"買いシグナル: {signal_counts.get(1, 0)}回")
print(f"売りシグナル: {signal_counts.get(-1, 0)}回")
print(f"無ポジション: {signal_counts.get(0, 0)}回")

# %% [markdown]
# ## バックテストの実行

# %%
# バックテストクラスの定義
class CryptoBacktester:
    def __init__(self, initial_capital, transaction_cost, stop_loss=None, take_profit=None):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
        # 状態変数
        self.capital = initial_capital
        self.position = 0  # 現在ポジション（コイン数）
        self.position_value = 0  # ポジション評価額
        self.entry_price = 0  # エントリー価格
        self.trades = []  # 取引履歴
        
        # パフォーマンストラッキング
        self.portfolio_values = []
        self.returns = []
        
    def execute_backtest(self, df):
        """バックテストの実行"""
        for i, (date, row) in enumerate(df.iterrows()):
            current_price = row['Close']
            signal = row['signal']
            
            # 現在のポートフォリオ価値計算
            if self.position > 0:
                self.position_value = self.position * current_price
                current_portfolio_value = self.capital + self.position_value
            else:
                current_portfolio_value = self.capital
            
            self.portfolio_values.append(current_portfolio_value)
            
            # シグナルに基づく取引実行
            if signal == 1 and self.position == 0:  # 買いシグナル
                self._execute_buy(current_price, date)
            elif signal == -1 and self.position > 0:  # 売りシグナル
                self._execute_sell(current_price, date)
            
            # ストップロス・テイクプロフィットのチェック
            if self.position > 0:
                self._check_stop_conditions(current_price, date)
                
        return self._calculate_performance()
    
    def _execute_buy(self, price, date):
        """買い注文の実行"""
        # 手数料を考慮した購入可能額
        available_capital = self.capital * (1 - self.transaction_cost)
        coins_to_buy = available_capital / price
        
        if coins_to_buy > 0:
            self.position = coins_to_buy
            self.entry_price = price
            self.capital = 0
            
            self.trades.append({
                'date': date,
                'action': 'BUY',
                'price': price,
                'quantity': coins_to_buy,
                'capital': self.capital + self.position * price
            })
    
    def _execute_sell(self, price, date):
        """売り注文の実行"""
        if self.position > 0:
            # 手数料を考慮した売却額
            proceeds = self.position * price * (1 - self.transaction_cost)
            
            self.capital = proceeds
            sold_quantity = self.position
            self.position = 0
            self.entry_price = 0
            
            self.trades.append({
                'date': date,
                'action': 'SELL',
                'price': price,
                'quantity': sold_quantity,
                'capital': self.capital
            })
    
    def _check_stop_conditions(self, current_price, date):
        """ストップロス・テイクプロフィットのチェック"""
        if self.position > 0 and self.entry_price > 0:
            price_change = (current_price - self.entry_price) / self.entry_price
            
            # ストップロス
            if self.stop_loss and price_change <= -self.stop_loss:
                self._execute_sell(current_price, date)
            
            # テイクプロフィット
            elif self.take_profit and price_change >= self.take_profit:
                self._execute_sell(current_price, date)
    
    def _calculate_performance(self):
        """パフォーマンス指標の計算"""
        if len(self.portfolio_values) == 0:
            return {}
        
        # リターンの計算
        portfolio_series = pd.Series(self.portfolio_values)
        returns = portfolio_series.pct_change().dropna()
        
        # パフォーマンス指標
        total_return = (portfolio_series.iloc[-1] / self.initial_capital) - 1
        annual_return = (1 + total_return) ** (252 / len(portfolio_series)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # 最大ドローダウン
        cumulative = portfolio_series / portfolio_series.expanding().max()
        max_drawdown = (cumulative.min() - 1)
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_capital': portfolio_series.iloc[-1],
            'num_trades': len(self.trades),
            'portfolio_values': self.portfolio_values,
            'trades': self.trades
        }

# %%
# バックテストの実行
print("バックテストを実行中...")

backtester = CryptoBacktester(
    initial_capital=INITIAL_CAPITAL,
    transaction_cost=TRANSACTION_COST,
    stop_loss=STOP_LOSS,
    take_profit=TAKE_PROFIT
)

performance = backtester.execute_backtest(df)

# 結果の表示
print(f"\n{'='*50}")
print(f"バックテスト結果 ({TARGET_SYMBOL})")
print(f"{'='*50}")
print(f"総リターン: {performance['total_return']:.2%}")
print(f"年率リターン: {performance['annual_return']:.2%}")
print(f"ボラティリティ: {performance['volatility']:.2%}")
print(f"シャープレシオ: {performance['sharpe_ratio']:.3f}")
print(f"最大ドローダウン: {performance['max_drawdown']:.2%}")
print(f"最終資金: ${performance['final_capital']:,.2f}")
print(f"取引回数: {performance['num_trades']}回")

# %% [markdown]
# ## ベンチマーク比較

# %%
# Buy & Hold戦略との比較
bh_returns = df['Close'].pct_change().dropna()
bh_cumulative = (1 + bh_returns).cumprod()
bh_total_return = bh_cumulative.iloc[-1] - 1
bh_annual_return = (1 + bh_total_return) ** (252 / len(bh_cumulative)) - 1
bh_volatility = bh_returns.std() * np.sqrt(252)
bh_sharpe = bh_annual_return / bh_volatility if bh_volatility > 0 else 0

# 最大ドローダウン（Buy & Hold）
bh_dd = (bh_cumulative / bh_cumulative.expanding().max()).min() - 1

print(f"\n{'='*50}")
print(f"ベンチマーク比較 (Buy & Hold)")
print(f"{'='*50}")
print(f"Buy & Hold総リターン: {bh_total_return:.2%}")
print(f"Buy & Hold年率リターン: {bh_annual_return:.2%}")
print(f"Buy & Holdボラティリティ: {bh_volatility:.2%}")
print(f"Buy & Holdシャープレシオ: {bh_sharpe:.3f}")
print(f"Buy & Hold最大ドローダウン: {bh_dd:.2%}")

print(f"\n{'='*30}")
print(f"戦略 vs ベンチマーク")
print(f"{'='*30}")
print(f"超過リターン: {(performance['total_return'] - bh_total_return):.2%}")
print(f"情報比率: {(performance['annual_return'] - bh_annual_return) / performance['volatility']:.3f}")

# %% [markdown]
# ## 結果の可視化

# %%
# パフォーマンスチャートの作成
fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# 1. ポートフォリオ価値の推移
portfolio_series = pd.Series(performance['portfolio_values'], index=df.index)
axes[0].plot(portfolio_series.index, portfolio_series.values, label='戦略', linewidth=2)
axes[0].plot(df.index, INITIAL_CAPITAL * bh_cumulative, label='Buy & Hold', linewidth=2)
axes[0].set_title('ポートフォリオ価値の推移')
axes[0].set_ylabel('価値 ($)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. 価格とシグナル
axes[1].plot(df.index, df['Close'], label='価格', alpha=0.7)
buy_signals = df[df['signal'] == 1]
sell_signals = df[df['signal'] == -1]
axes[1].scatter(buy_signals.index, buy_signals['Close'], color='green', marker='^', s=50, label='買いシグナル')
axes[1].scatter(sell_signals.index, sell_signals['Close'], color='red', marker='v', s=50, label='売りシグナル')
axes[1].set_title(f'{TARGET_SYMBOL} 価格と取引シグナル')
axes[1].set_ylabel('価格 (USDT)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3. 予測確率
axes[2].plot(df.index, df['prediction_prob'], label='予測確率', alpha=0.7)
axes[2].axhline(y=MIN_CONFIDENCE, color='green', linestyle='--', alpha=0.5, label=f'買い閾値 ({MIN_CONFIDENCE})')
axes[2].axhline(y=1-MIN_CONFIDENCE, color='red', linestyle='--', alpha=0.5, label=f'売り閾値 ({1-MIN_CONFIDENCE})')
axes[2].set_title('モデル予測確率')
axes[2].set_ylabel('確率')
axes[2].set_xlabel('日付')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 取引分析

# %%
# 取引履歴の分析
if performance['trades']:
    trades_df = pd.DataFrame(performance['trades'])
    
    print(f"\n取引履歴詳細:")
    print(trades_df.head(10))
    
    # 勝率の計算
    if len(trades_df) >= 2:
        buy_trades = trades_df[trades_df['action'] == 'BUY']
        sell_trades = trades_df[trades_df['action'] == 'SELL']
        
        if len(buy_trades) > 0 and len(sell_trades) > 0:
            # ペアの取引を特定
            profitable_trades = 0
            total_pairs = min(len(buy_trades), len(sell_trades))
            
            for i in range(total_pairs):
                buy_price = buy_trades.iloc[i]['price']
                sell_price = sell_trades.iloc[i]['price']
                if sell_price > buy_price:
                    profitable_trades += 1
            
            win_rate = profitable_trades / total_pairs if total_pairs > 0 else 0
            print(f"\n勝率: {win_rate:.2%} ({profitable_trades}/{total_pairs})")
    
    # 平均取引間隔
    if len(trades_df) > 1:
        trade_dates = pd.to_datetime(trades_df['date'])
        avg_interval = (trade_dates.max() - trade_dates.min()) / len(trades_df)
        print(f"平均取引間隔: {avg_interval.days}日")

# %% [markdown]
# ## リスク分析

# %%
# VaR（Value at Risk）の計算
if len(performance['portfolio_values']) > 1:
    portfolio_returns = pd.Series(performance['portfolio_values']).pct_change().dropna()
    
    # VaR (95%, 99%)
    var_95 = np.percentile(portfolio_returns, 5)
    var_99 = np.percentile(portfolio_returns, 1)
    
    print(f"\nリスク指標:")
    print(f"VaR (95%): {var_95:.2%}")
    print(f"VaR (99%): {var_99:.2%}")
    
    # 最大連続損失日数
    losses = portfolio_returns < 0
    if losses.any():
        loss_streaks = []
        current_streak = 0
        
        for is_loss in losses:
            if is_loss:
                current_streak += 1
            else:
                if current_streak > 0:
                    loss_streaks.append(current_streak)
                current_streak = 0
        
        if current_streak > 0:
            loss_streaks.append(current_streak)
        
        max_loss_streak = max(loss_streaks) if loss_streaks else 0
        print(f"最大連続損失日数: {max_loss_streak}日")

# %% [markdown]
# ## パラメータ最適化

# %%
# 信頼度閾値の最適化
print("\n信頼度閾値の最適化を実行中...")

confidence_levels = np.arange(0.5, 0.9, 0.05)
optimization_results = []

for confidence in confidence_levels:
    # 一時的なシグナル生成
    temp_df = df.copy()
    temp_df['signal'] = 0
    
    buy_condition = (temp_df['prediction_prob'] > confidence)
    sell_condition = (temp_df['prediction_prob'] < (1 - confidence))
    
    temp_df.loc[buy_condition, 'signal'] = 1
    temp_df.loc[sell_condition, 'signal'] = -1
    
    # バックテスト実行
    temp_backtester = CryptoBacktester(
        initial_capital=INITIAL_CAPITAL,
        transaction_cost=TRANSACTION_COST,
        stop_loss=STOP_LOSS,
        take_profit=TAKE_PROFIT
    )
    
    temp_performance = temp_backtester.execute_backtest(temp_df)
    
    optimization_results.append({
        'confidence': confidence,
        'total_return': temp_performance['total_return'],
        'sharpe_ratio': temp_performance['sharpe_ratio'],
        'max_drawdown': temp_performance['max_drawdown'],
        'num_trades': temp_performance['num_trades']
    })

opt_df = pd.DataFrame(optimization_results)
best_confidence = opt_df.loc[opt_df['sharpe_ratio'].idxmax(), 'confidence']

print(f"最適信頼度閾値: {best_confidence:.2f}")
print(f"最適化結果:")
print(opt_df.round(4))

# 最適化結果の可視化
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0,0].plot(opt_df['confidence'], opt_df['total_return'])
axes[0,0].set_title('総リターン vs 信頼度閾値')
axes[0,0].set_xlabel('信頼度閾値')
axes[0,0].set_ylabel('総リターン')
axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(opt_df['confidence'], opt_df['sharpe_ratio'])
axes[0,1].set_title('シャープレシオ vs 信頼度閾値')
axes[0,1].set_xlabel('信頼度閾値')
axes[0,1].set_ylabel('シャープレシオ')
axes[0,1].grid(True, alpha=0.3)

axes[1,0].plot(opt_df['confidence'], opt_df['max_drawdown'])
axes[1,0].set_title('最大ドローダウン vs 信頼度閾値')
axes[1,0].set_xlabel('信頼度閾値')
axes[1,0].set_ylabel('最大ドローダウン')
axes[1,0].grid(True, alpha=0.3)

axes[1,1].plot(opt_df['confidence'], opt_df['num_trades'])
axes[1,1].set_title('取引回数 vs 信頼度閾値')
axes[1,1].set_xlabel('信頼度閾値')
axes[1,1].set_ylabel('取引回数')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 結果の保存

# %%
# バックテスト結果を保存
backtest_results = {
    'performance': performance,
    'benchmark': {
        'total_return': bh_total_return,
        'annual_return': bh_annual_return,
        'volatility': bh_volatility,
        'sharpe_ratio': bh_sharpe,
        'max_drawdown': bh_dd
    },
    'optimization': opt_df.to_dict('records'),
    'best_confidence': best_confidence,
    'model_name': best_model_name,
    'parameters': {
        'initial_capital': INITIAL_CAPITAL,
        'transaction_cost': TRANSACTION_COST,
        'stop_loss': STOP_LOSS,
        'take_profit': TAKE_PROFIT,
        'min_confidence': MIN_CONFIDENCE
    }
}

save_data(backtest_results, 'backtest_results.pkl')
print("バックテスト結果を保存しました: backtest_results.pkl")

# CSVでも保存
if performance['trades']:
    trades_df = pd.DataFrame(performance['trades'])
    save_data(trades_df, 'trades_history.csv')
    print("取引履歴を保存しました: trades_history.csv")

# %% [markdown]
# ## まとめ
#
# ### 完了したタスク
# 1. ✅ 学習済みモデルの読み込み
# 2. ✅ 取引シグナルの生成
# 3. ✅ バックテストの実行
# 4. ✅ ベンチマーク比較
# 5. ✅ パフォーマンス分析・可視化
# 6. ✅ リスク評価
# 7. ✅ パラメータ最適化
#
# ### バックテスト結果サマリー
# - 使用モデル: {best_model_name}
# - 総リターン: {performance['total_return']:.2%}
# - 年率リターン: {performance['annual_return']:.2%}
# - シャープレシオ: {performance['sharpe_ratio']:.3f}
# - 最大ドローダウン: {performance['max_drawdown']:.2%}
# - 取引回数: {performance['num_trades']}回
# - ベンチマーク超過リターン: {(performance['total_return'] - bh_total_return):.2%}
#
# ### 出力ファイル
# - `backtest_results.pkl`: 詳細なバックテスト結果
# - `trades_history.csv`: 取引履歴
#
# ### 推奨事項
# 1. 最適信頼度閾値 {best_confidence:.2f} の使用を検討
# 2. リスク管理パラメータの定期的な見直し
# 3. 市場環境変化に応じたモデル再学習の実施