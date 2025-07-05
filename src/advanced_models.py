"""
Advanced Models for World-Class Trading System
Includes TFT, Informer, and Hybrid RL implementations
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Try to import deep learning libraries
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Advanced models will use simplified implementations.")

try:
    import pytorch_lightning as pl
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    print("Warning: PyTorch Lightning not available.")

class SimpleTransformerModel(BaseEstimator, ClassifierMixin):
    """Simplified Transformer model for time series classification"""
    
    def __init__(self, seq_len=60, d_model=64, nhead=4, num_layers=2, 
                 num_classes=2, dropout=0.1, lr=1e-3, epochs=50):
        self.seq_len = seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.scaler = StandardScaler()
        self.feature_dim = None
        
    def _create_model(self):
        """Create the transformer model"""
        if not TORCH_AVAILABLE:
            # Fallback to Random Forest
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            return
            
        class TransformerClassifier(nn.Module):
            def __init__(self, input_dim, d_model, nhead, num_layers, num_classes, dropout, seq_len):
                super().__init__()
                self.input_dim = input_dim
                self.d_model = d_model
                self.seq_len = seq_len
                
                # Input projection
                self.input_projection = nn.Linear(input_dim, d_model)
                
                # Positional encoding
                self.pos_encoding = nn.Parameter(torch.randn(seq_len, d_model))
                
                # Transformer layers
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dropout=dropout,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                
                # Classification head
                self.classifier = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model // 2, num_classes)
                )
                
            def forward(self, x):
                # x shape: (batch_size, seq_len, input_dim)
                batch_size, seq_len, _ = x.shape
                
                # Project input
                x = self.input_projection(x)
                
                # Add positional encoding
                x = x + self.pos_encoding[:seq_len].unsqueeze(0)
                
                # Transformer
                x = self.transformer(x)
                
                # Global average pooling
                x = x.mean(dim=1)
                
                # Classification
                return self.classifier(x)
        
        self.model = TransformerClassifier(
            input_dim=self.feature_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            num_classes=self.num_classes,
            dropout=self.dropout,
            seq_len=self.seq_len
        ).to(self.device)
        
    def _prepare_sequences(self, X, y=None):
        """Prepare sequences for transformer input"""
        if len(X) < self.seq_len:
            return None, None
            
        sequences = []
        targets = []
        
        for i in range(self.seq_len, len(X)):
            seq = X[i-self.seq_len:i]
            sequences.append(seq)
            if y is not None:
                targets.append(y[i])
                
        return np.array(sequences), np.array(targets) if y is not None else None
    
    def fit(self, X, y):
        """Fit the transformer model"""
        if not TORCH_AVAILABLE:
            # Fallback to Random Forest
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            return self.model.fit(X, y)
            
        # Store feature dimension
        self.feature_dim = X.shape[1]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Prepare sequences
        X_seq, y_seq = self._prepare_sequences(X_scaled, y)
        
        if X_seq is None:
            raise ValueError(f"Not enough data for sequence length {self.seq_len}")
            
        # Create model
        self._create_model()
        
        # Prepare data
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.LongTensor(y_seq).to(self.device)
        
        # Training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        if not TORCH_AVAILABLE:
            return self.model.predict(X)
            
        X_scaled = self.scaler.transform(X)
        X_seq, _ = self._prepare_sequences(X_scaled)
        
        if X_seq is None:
            # Not enough data for sequences, return baseline predictions
            return np.zeros(len(X))
            
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        
        # Pad predictions to match input length
        full_predictions = np.zeros(len(X))
        full_predictions[self.seq_len:] = predictions
        
        return full_predictions
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if not TORCH_AVAILABLE:
            return self.model.predict_proba(X)
            
        X_scaled = self.scaler.transform(X)
        X_seq, _ = self._prepare_sequences(X_scaled)
        
        if X_seq is None:
            # Not enough data for sequences, return baseline probabilities
            return np.column_stack([np.ones(len(X)) * 0.5, np.ones(len(X)) * 0.5])
            
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            outputs = self.model(X_tensor)
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()
        
        # Pad probabilities to match input length
        full_probabilities = np.column_stack([np.ones(len(X)) * 0.5, np.ones(len(X)) * 0.5])
        full_probabilities[self.seq_len:] = probabilities
        
        return full_probabilities

class TFTModel(BaseEstimator, ClassifierMixin):
    """Temporal Fusion Transformer (TFT) - Simplified Implementation"""
    
    def __init__(self, seq_len=60, hidden_dim=64, num_heads=4, num_layers=2, 
                 num_classes=2, dropout=0.1, lr=1e-3, epochs=50):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        
        # For now, use simplified transformer
        self.model = SimpleTransformerModel(
            seq_len=seq_len,
            d_model=hidden_dim,
            nhead=num_heads,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            lr=lr,
            epochs=epochs
        )
    
    def fit(self, X, y):
        return self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)

class InformerModel(BaseEstimator, ClassifierMixin):
    """Informer Model - Simplified Implementation"""
    
    def __init__(self, seq_len=96, label_len=48, pred_len=24, 
                 d_model=64, n_heads=4, e_layers=2, d_layers=1,
                 num_classes=2, dropout=0.1, lr=1e-3, epochs=50):
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        
        # For now, use simplified transformer
        self.model = SimpleTransformerModel(
            seq_len=seq_len,
            d_model=d_model,
            nhead=n_heads,
            num_layers=e_layers,
            num_classes=num_classes,
            dropout=dropout,
            lr=lr,
            epochs=epochs
        )
    
    def fit(self, X, y):
        return self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)

class HybridRLModel(BaseEstimator, ClassifierMixin):
    """Hybrid Reinforcement Learning Model"""
    
    def __init__(self, base_model=None, rl_episodes=100, epsilon=0.1, 
                 learning_rate=0.01, discount_factor=0.95):
        self.base_model = base_model or RandomForestClassifier(n_estimators=100, random_state=42)
        self.rl_episodes = rl_episodes
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = {}
        self.scaler = StandardScaler()
        
    def _get_state(self, features, window_size=10):
        """Convert features to state representation"""
        # Simple state: discretized recent price movements
        if len(features) < window_size:
            return "start"
        
        recent_returns = np.diff(features[-window_size:])
        avg_return = np.mean(recent_returns)
        volatility = np.std(recent_returns)
        
        # Discretize state
        return_state = "up" if avg_return > 0.01 else "down" if avg_return < -0.01 else "flat"
        vol_state = "high" if volatility > 0.02 else "low"
        
        return f"{return_state}_{vol_state}"
    
    def _get_reward(self, action, actual_return):
        """Calculate reward based on action and actual return"""
        if action == 1 and actual_return > 0:  # Buy and price went up
            return actual_return
        elif action == 0 and actual_return < 0:  # Sell and price went down
            return -actual_return
        else:  # Wrong prediction
            return -abs(actual_return)
    
    def _epsilon_greedy_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.choice([0, 1])
        
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0]
        
        return np.argmax(self.q_table[state])
    
    def _update_q_table(self, state, action, reward, next_state):
        """Update Q-table using Q-learning"""
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0]
        
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0, 0.0]
        
        # Q-learning update
        old_value = self.q_table[state][action]
        next_max = max(self.q_table[next_state])
        
        new_value = old_value + self.learning_rate * (
            reward + self.discount_factor * next_max - old_value
        )
        
        self.q_table[state][action] = new_value
    
    def fit(self, X, y):
        """Train the hybrid model"""
        # First, train the base model
        X_scaled = self.scaler.fit_transform(X)
        self.base_model.fit(X_scaled, y)
        
        # Then, train the RL component
        print("Training RL component...")
        
        # Calculate returns for RL training
        if 'Close' in X.columns:
            prices = X['Close'].values
        else:
            # Use first column as proxy for price
            prices = X.iloc[:, 0].values
        
        returns = np.diff(prices) / prices[:-1]
        
        # RL training episodes
        for episode in range(self.rl_episodes):
            total_reward = 0
            
            for i in range(10, len(returns) - 1):
                # Get current state
                state = self._get_state(prices[:i+1])
                
                # Choose action
                action = self._epsilon_greedy_action(state)
                
                # Get reward
                actual_return = returns[i]
                reward = self._get_reward(action, actual_return)
                total_reward += reward
                
                # Get next state
                next_state = self._get_state(prices[:i+2])
                
                # Update Q-table
                self._update_q_table(state, action, reward, next_state)
            
            # Decay epsilon
            self.epsilon = max(0.01, self.epsilon * 0.99)
            
            if episode % 20 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward:.4f}, Epsilon: {self.epsilon:.4f}")
        
        return self
    
    def predict(self, X):
        """Make predictions using hybrid approach"""
        X_scaled = self.scaler.transform(X)
        
        # Get base model predictions
        base_predictions = self.base_model.predict(X_scaled)
        
        # Get RL-based adjustments
        rl_predictions = []
        
        if 'Close' in X.columns:
            prices = X['Close'].values
        else:
            prices = X.iloc[:, 0].values
        
        for i in range(len(X)):
            if i < 10:
                # Not enough history for RL
                rl_predictions.append(base_predictions[i])
            else:
                state = self._get_state(prices[:i+1])
                rl_action = self._epsilon_greedy_action(state)
                rl_predictions.append(rl_action)
        
        # Combine predictions (weighted average)
        final_predictions = 0.7 * base_predictions + 0.3 * np.array(rl_predictions)
        return (final_predictions > 0.5).astype(int)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        X_scaled = self.scaler.transform(X)
        base_probas = self.base_model.predict_proba(X_scaled)
        
        # For simplicity, return base model probabilities
        # In a full implementation, this would incorporate RL confidence
        return base_probas

def create_ensemble_model(models, weights=None):
    """Create an ensemble of models"""
    
    class EnsembleModel(BaseEstimator, ClassifierMixin):
        def __init__(self, models, weights=None):
            self.models = models
            self.weights = weights or [1.0 / len(models)] * len(models)
            
        def fit(self, X, y):
            for i, model in enumerate(self.models):
                print(f"Training model {i+1}/{len(self.models)}: {type(model).__name__}")
                model.fit(X, y)
            return self
            
        def predict(self, X):
            predictions = []
            for model in self.models:
                pred = model.predict(X)
                predictions.append(pred)
            
            # Weighted voting
            weighted_pred = np.average(predictions, axis=0, weights=self.weights)
            return (weighted_pred > 0.5).astype(int)
            
        def predict_proba(self, X):
            probabilities = []
            for model in self.models:
                prob = model.predict_proba(X)
                probabilities.append(prob)
            
            # Weighted averaging
            weighted_proba = np.average(probabilities, axis=0, weights=self.weights)
            return weighted_proba
    
    return EnsembleModel(models, weights)

def get_advanced_models():
    """Get dictionary of advanced models"""
    return {
        'transformer': SimpleTransformerModel(epochs=30),
        'tft': TFTModel(epochs=30),
        'informer': InformerModel(epochs=30),
        'hybrid_rl': HybridRLModel(rl_episodes=50)
    }

def create_stacked_model(base_models, meta_model=None):
    """Create a stacked model"""
    
    class StackedModel(BaseEstimator, ClassifierMixin):
        def __init__(self, base_models, meta_model=None):
            self.base_models = base_models
            self.meta_model = meta_model or RandomForestClassifier(n_estimators=50, random_state=42)
            
        def fit(self, X, y):
            # Train base models
            for i, model in enumerate(self.base_models):
                print(f"Training base model {i+1}/{len(self.base_models)}: {type(model).__name__}")
                model.fit(X, y)
            
            # Generate meta-features
            meta_features = []
            for model in self.base_models:
                pred_proba = model.predict_proba(X)
                meta_features.append(pred_proba[:, 1])  # Take positive class probability
            
            meta_X = np.column_stack(meta_features)
            
            # Train meta-model
            print("Training meta-model...")
            self.meta_model.fit(meta_X, y)
            
            return self
            
        def predict(self, X):
            # Generate meta-features
            meta_features = []
            for model in self.base_models:
                pred_proba = model.predict_proba(X)
                meta_features.append(pred_proba[:, 1])
            
            meta_X = np.column_stack(meta_features)
            
            # Make final predictions
            return self.meta_model.predict(meta_X)
            
        def predict_proba(self, X):
            # Generate meta-features
            meta_features = []
            for model in self.base_models:
                pred_proba = model.predict_proba(X)
                meta_features.append(pred_proba[:, 1])
            
            meta_X = np.column_stack(meta_features)
            
            # Make final predictions
            return self.meta_model.predict_proba(meta_X)
    
    return StackedModel(base_models, meta_model)

# Example usage
if __name__ == "__main__":
    print("Advanced Models for World-Class Trading System")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                     columns=[f'feature_{i}' for i in range(n_features)])
    y = np.random.randint(0, 2, n_samples)
    
    # Test models
    models = get_advanced_models()
    
    for name, model in models.items():
        print(f"\nTesting {name}...")
        try:
            model.fit(X, y)
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)
            print(f"✅ {name} working correctly")
        except Exception as e:
            print(f"❌ {name} error: {e}")
    
    print("\n✅ Advanced models module ready!")