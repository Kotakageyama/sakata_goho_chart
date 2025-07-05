"""
Advanced Feature Engineering for World-Class Model
Implements TEMA, Keltner Channel, NATR, ATR-bands, tsfresh features, 
Hurst exponent, and variance ratio analysis.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

try:
    import tsfresh
    from tsfresh import extract_features
    from tsfresh.feature_extraction import EfficientFCParameters
    TSFRESH_AVAILABLE = True
except ImportError:
    TSFRESH_AVAILABLE = False
    print("Warning: tsfresh not installed. Time series features will be limited.")

def tema(data, period=14):
    """Triple Exponential Moving Average (TEMA)"""
    ema1 = data.ewm(span=period).mean()
    ema2 = ema1.ewm(span=period).mean()
    ema3 = ema2.ewm(span=period).mean()
    return 3 * ema1 - 3 * ema2 + ema3

def keltner_channel(high, low, close, period=20, multiplier=2.0):
    """Keltner Channel implementation"""
    ema = close.ewm(span=period).mean()
    tr = np.maximum(high - low, 
                    np.maximum(abs(high - close.shift(1)), 
                              abs(low - close.shift(1))))
    atr = tr.rolling(window=period).mean()
    
    upper = ema + multiplier * atr
    lower = ema - multiplier * atr
    
    return upper, lower, ema

def natr(high, low, close, period=14):
    """Normalized Average True Range (NATR)"""
    tr = np.maximum(high - low, 
                    np.maximum(abs(high - close.shift(1)), 
                              abs(low - close.shift(1))))
    atr = tr.rolling(window=period).mean()
    return (atr / close) * 100

def atr_bands(high, low, close, period=14, multiplier=2.0):
    """ATR-based bands"""
    tr = np.maximum(high - low, 
                    np.maximum(abs(high - close.shift(1)), 
                              abs(low - close.shift(1))))
    atr = tr.rolling(window=period).mean()
    
    ema = close.ewm(span=period).mean()
    upper_band = ema + multiplier * atr
    lower_band = ema - multiplier * atr
    
    return upper_band, lower_band

def hurst_exponent(ts, max_lag=100):
    """Calculate Hurst exponent for fractal analysis"""
    if len(ts) < 2:
        return 0.5
    
    # Remove NaN values
    ts = ts.dropna()
    if len(ts) < 2:
        return 0.5
    
    lags = range(2, min(max_lag, len(ts) // 2))
    tau = []
    
    for lag in lags:
        # Calculate log returns
        returns = np.diff(np.log(ts))
        
        # Calculate variance for different lags
        pp = 0
        for i in range(lag, len(returns)):
            pp += (returns[i] - returns[i-lag]) ** 2
        
        pp = pp / (len(returns) - lag)
        tau.append(pp)
    
    # Linear regression to find Hurst exponent
    if len(tau) > 1:
        tau = np.array(tau)
        tau = tau[tau > 0]  # Remove zeros
        if len(tau) > 1:
            log_tau = np.log(tau)
            log_lags = np.log(lags[:len(tau)])
            
            # Simple linear regression
            n = len(log_lags)
            if n > 1:
                slope = (n * np.sum(log_lags * log_tau) - np.sum(log_lags) * np.sum(log_tau)) / \
                       (n * np.sum(log_lags**2) - np.sum(log_lags)**2)
                hurst = slope / 2
                return max(0, min(1, hurst))
    
    return 0.5

def variance_ratio(ts, lags=[2, 4, 8, 16]):
    """Calculate variance ratio for different lags"""
    if len(ts) < max(lags) + 1:
        return {f'vr_{lag}': 1.0 for lag in lags}
    
    # Calculate log returns
    returns = np.diff(np.log(ts.dropna()))
    
    # Base variance (lag 1)
    base_var = np.var(returns)
    
    if base_var == 0:
        return {f'vr_{lag}': 1.0 for lag in lags}
    
    vr_dict = {}
    for lag in lags:
        if len(returns) >= lag:
            # Calculate variance for lag k
            lag_returns = []
            for i in range(lag, len(returns)):
                lag_returns.append(sum(returns[i-lag:i]))
            
            if len(lag_returns) > 1:
                lag_var = np.var(lag_returns)
                vr_dict[f'vr_{lag}'] = lag_var / (lag * base_var) if base_var > 0 else 1.0
            else:
                vr_dict[f'vr_{lag}'] = 1.0
        else:
            vr_dict[f'vr_{lag}'] = 1.0
    
    return vr_dict

def create_autoencoder_features(df, window=128, latent_dim=16):
    """Create AutoEncoder latent features (simplified version)"""
    # For now, use PCA as a proxy for AutoEncoder features
    # In a full implementation, this would use TensorFlow/PyTorch
    
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    available_cols = [col for col in feature_cols if col in df.columns]
    
    if len(available_cols) == 0:
        return pd.DataFrame(index=df.index)
    
    # Create rolling windows
    windowed_features = []
    for i in range(window, len(df)):
        window_data = df[available_cols].iloc[i-window:i].values.flatten()
        windowed_features.append(window_data)
    
    if len(windowed_features) == 0:
        return pd.DataFrame(index=df.index)
    
    # Apply PCA (proxy for AutoEncoder)
    windowed_features = np.array(windowed_features)
    
    # Handle NaN values
    windowed_features = np.nan_to_num(windowed_features)
    
    # Standardize
    scaler = StandardScaler()
    windowed_features_scaled = scaler.fit_transform(windowed_features)
    
    # Apply PCA
    pca = PCA(n_components=min(latent_dim, windowed_features_scaled.shape[1]))
    latent_features = pca.fit_transform(windowed_features_scaled)
    
    # Create DataFrame
    latent_df = pd.DataFrame(
        latent_features,
        columns=[f'latent_{i}' for i in range(latent_features.shape[1])],
        index=df.index[window:]
    )
    
    # Align with original DataFrame
    result_df = pd.DataFrame(index=df.index)
    for col in latent_df.columns:
        result_df[col] = np.nan
        result_df.loc[latent_df.index, col] = latent_df[col]
    
    return result_df

def extract_tsfresh_features(df, window=50):
    """Extract tsfresh features from time series data"""
    if not TSFRESH_AVAILABLE:
        return pd.DataFrame(index=df.index)
    
    features_list = []
    
    # Use efficient feature extraction settings
    settings = EfficientFCParameters()
    
    for i in range(window, len(df)):
        try:
            # Create time series for the window
            ts_data = df[['Close']].iloc[i-window:i].reset_index()
            ts_data['id'] = 1
            ts_data['time'] = range(len(ts_data))
            
            # Extract features
            features = extract_features(
                ts_data[['id', 'time', 'Close']], 
                column_id='id', 
                column_sort='time',
                default_fc_parameters=settings
            )
            
            features_list.append(features.iloc[0])
            
        except Exception as e:
            # If extraction fails, append NaN row
            if len(features_list) > 0:
                features_list.append(pd.Series(index=features_list[-1].index, dtype=float))
            else:
                features_list.append(pd.Series(dtype=float))
    
    if len(features_list) == 0:
        return pd.DataFrame(index=df.index)
    
    # Combine features
    tsfresh_df = pd.DataFrame(features_list, index=df.index[window:])
    
    # Align with original DataFrame
    result_df = pd.DataFrame(index=df.index)
    for col in tsfresh_df.columns:
        result_df[col] = np.nan
        result_df.loc[tsfresh_df.index, col] = tsfresh_df[col]
    
    return result_df

def add_advanced_features(df):
    """Add all advanced features to the dataframe"""
    df_enhanced = df.copy()
    
    # Required columns check
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    available_cols = [col for col in required_cols if col in df.columns]
    
    if 'Close' not in available_cols:
        print("Warning: 'Close' column not found. Cannot add advanced features.")
        return df_enhanced
    
    print("Adding advanced technical indicators...")
    
    # Advanced Technical Indicators
    if all(col in df.columns for col in ['High', 'Low', 'Close']):
        # TEMA
        df_enhanced['tema_14'] = tema(df['Close'], 14)
        df_enhanced['tema_21'] = tema(df['Close'], 21)
        
        # Keltner Channel
        kc_upper, kc_lower, kc_mid = keltner_channel(df['High'], df['Low'], df['Close'])
        df_enhanced['kc_upper'] = kc_upper
        df_enhanced['kc_lower'] = kc_lower
        df_enhanced['kc_mid'] = kc_mid
        df_enhanced['kc_width'] = (kc_upper - kc_lower) / kc_mid
        
        # NATR
        df_enhanced['natr_14'] = natr(df['High'], df['Low'], df['Close'], 14)
        df_enhanced['natr_21'] = natr(df['High'], df['Low'], df['Close'], 21)
        
        # ATR Bands
        atr_upper, atr_lower = atr_bands(df['High'], df['Low'], df['Close'])
        df_enhanced['atr_upper'] = atr_upper
        df_enhanced['atr_lower'] = atr_lower
        df_enhanced['atr_position'] = (df['Close'] - atr_lower) / (atr_upper - atr_lower)
    
    print("Adding fractal characteristics...")
    
    # Fractal Characteristics
    rolling_window = 50
    df_enhanced['hurst_exp'] = df['Close'].rolling(window=rolling_window).apply(
        lambda x: hurst_exponent(x), raw=False
    )
    
    # Variance Ratio
    vr_results = df['Close'].rolling(window=rolling_window).apply(
        lambda x: pd.Series(variance_ratio(x)), raw=False
    )
    
    if isinstance(vr_results, pd.Series):
        # Handle case where apply returns a Series
        for lag in [2, 4, 8, 16]:
            df_enhanced[f'vr_{lag}'] = np.nan
    else:
        # Handle case where apply returns a DataFrame
        try:
            for lag in [2, 4, 8, 16]:
                col_name = f'vr_{lag}'
                if col_name in vr_results.columns:
                    df_enhanced[col_name] = vr_results[col_name]
                else:
                    df_enhanced[col_name] = np.nan
        except:
            for lag in [2, 4, 8, 16]:
                df_enhanced[f'vr_{lag}'] = np.nan
    
    print("Adding time series representation learning features...")
    
    # AutoEncoder Features (using PCA as proxy)
    try:
        ae_features = create_autoencoder_features(df, window=64, latent_dim=8)
        df_enhanced = pd.concat([df_enhanced, ae_features], axis=1)
    except Exception as e:
        print(f"Warning: Could not create AutoEncoder features: {e}")
    
    # tsfresh Features (subset for performance)
    try:
        if TSFRESH_AVAILABLE:
            tsfresh_features = extract_tsfresh_features(df, window=30)
            # Limit to most important features to avoid memory issues
            if len(tsfresh_features.columns) > 20:
                tsfresh_features = tsfresh_features.iloc[:, :20]
            df_enhanced = pd.concat([df_enhanced, tsfresh_features], axis=1)
    except Exception as e:
        print(f"Warning: Could not extract tsfresh features: {e}")
    
    print("Adding market microstructure features...")
    
    # Market Microstructure Features
    if 'Volume' in df.columns:
        df_enhanced['volume_sma_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df_enhanced['price_volume_trend'] = (df['Close'] - df['Close'].shift(1)) * df['Volume']
        df_enhanced['volume_price_ratio'] = df['Volume'] / df['Close']
    
    # Price action features
    df_enhanced['price_momentum'] = df['Close'].pct_change(5)
    df_enhanced['price_acceleration'] = df_enhanced['price_momentum'].diff()
    
    # Gap features
    if 'Open' in df.columns:
        df_enhanced['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        df_enhanced['gap_filled'] = ((df['High'] >= df['Close'].shift(1)) & 
                                   (df['Low'] <= df['Close'].shift(1))).astype(int)
    
    print(f"Advanced features added. Total features: {len(df_enhanced.columns)}")
    
    return df_enhanced

def get_feature_importance_categories():
    """Return feature categories for analysis"""
    return {
        'technical_advanced': ['tema_14', 'tema_21', 'kc_upper', 'kc_lower', 'kc_mid', 
                              'kc_width', 'natr_14', 'natr_21', 'atr_upper', 'atr_lower', 
                              'atr_position'],
        'fractal': ['hurst_exp', 'vr_2', 'vr_4', 'vr_8', 'vr_16'],
        'autoencoder': [f'latent_{i}' for i in range(8)],
        'microstructure': ['volume_sma_ratio', 'price_volume_trend', 'volume_price_ratio',
                          'price_momentum', 'price_acceleration', 'gap', 'gap_filled']
    }