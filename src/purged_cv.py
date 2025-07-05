"""
Purged Walk-Forward Cross-Validation Implementation
Based on Marcos López de Prado's methodology for time series validation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')

class PurgedWalkForwardCV(BaseCrossValidator):
    """
    Purged Walk-Forward Cross-Validation
    
    This implementation addresses the data leakage issues in time series
    by implementing purging and embargoing techniques.
    
    Parameters:
    -----------
    n_splits : int, default=5
        Number of splits for cross-validation
    max_train_size : int, default=None
        Maximum size of the training set
    test_size : int, default=None
        Size of the test set
    purge_size : int, default=0
        Number of samples to purge after training set
    embargo_size : int, default=0
        Number of samples to embargo after test set
    """
    
    def __init__(self, n_splits=5, max_train_size=None, test_size=None, 
                 purge_size=0, embargo_size=0):
        self.n_splits = n_splits
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.purge_size = purge_size
        self.embargo_size = embargo_size
        
    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set."""
        n_samples = X.shape[0]
        
        # Calculate test size
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size
            
        # Generate splits
        indices = np.arange(n_samples)
        
        for i in range(self.n_splits):
            # Calculate test start and end
            test_start = (i + 1) * test_size + i * (self.purge_size + self.embargo_size)
            test_end = test_start + test_size
            
            # Ensure we don't exceed array bounds
            if test_end > n_samples:
                break
                
            # Training set ends before purge
            train_end = test_start - self.purge_size
            
            # Training set starts
            if self.max_train_size is None:
                train_start = 0
            else:
                train_start = max(0, train_end - self.max_train_size)
                
            # Ensure valid indices
            if train_start >= train_end or train_end <= 0:
                continue
                
            # Create training and test indices
            train_indices = indices[train_start:train_end]
            test_indices = indices[test_start:test_end]
            
            yield train_indices, test_indices
            
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations."""
        return self.n_splits

def purged_cv_score(estimator, X, y, cv=None, scoring=None, n_jobs=None, 
                   verbose=0, fit_params=None, pre_dispatch='2*n_jobs'):
    """
    Purged Cross-Validation Score
    
    Parameters:
    -----------
    estimator : estimator object
        The object to use to fit the data
    X : array-like, shape = [n_samples, n_features]
        The data to fit
    y : array-like, shape = [n_samples]
        The target variable
    cv : cross-validation generator
        Cross-validation splitting strategy
    scoring : string, callable or None
        Scoring method
    
    Returns:
    --------
    scores : array of float, shape = [n_splits]
        Array of scores of the estimator for each run
    """
    
    if cv is None:
        cv = PurgedWalkForwardCV()
        
    scores = []
    
    for train_idx, test_idx in cv.split(X, y):
        # Clone the estimator
        estimator_clone = clone(estimator)
        
        # Fit on training data
        X_train = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
        y_train = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
        
        X_test = X.iloc[test_idx] if hasattr(X, 'iloc') else X[test_idx]
        y_test = y.iloc[test_idx] if hasattr(y, 'iloc') else y[test_idx]
        
        # Fit the model
        estimator_clone.fit(X_train, y_train)
        
        # Make predictions
        y_pred = estimator_clone.predict(X_test)
        
        # Calculate score
        if scoring is None:
            score = estimator_clone.score(X_test, y_test)
        elif hasattr(scoring, '__call__'):
            score = scoring(y_test, y_pred)
        else:
            # Handle string scoring
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
            if scoring == 'accuracy':
                score = accuracy_score(y_test, y_pred)
            elif scoring == 'f1':
                score = f1_score(y_test, y_pred, average='weighted')
            elif scoring == 'roc_auc':
                y_pred_proba = estimator_clone.predict_proba(X_test)[:, 1]
                score = roc_auc_score(y_test, y_pred_proba)
            else:
                score = accuracy_score(y_test, y_pred)
        
        scores.append(score)
        
        if verbose > 0:
            print(f"Fold score: {score:.4f}")
    
    return np.array(scores)

def calculate_cv_metrics(cv_scores):
    """Calculate comprehensive CV metrics"""
    return {
        'mean_score': np.mean(cv_scores),
        'std_score': np.std(cv_scores),
        'min_score': np.min(cv_scores),
        'max_score': np.max(cv_scores),
        'median_score': np.median(cv_scores),
        'cv_score_range': np.max(cv_scores) - np.min(cv_scores),
        'stability_ratio': np.std(cv_scores) / np.mean(cv_scores) if np.mean(cv_scores) != 0 else 0
    }

def plot_cv_results(cv_scores, cv_dates=None, title="Cross-Validation Results"):
    """Plot cross-validation results"""
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot scores over time
        x_vals = cv_dates if cv_dates is not None else range(len(cv_scores))
        ax1.plot(x_vals, cv_scores, 'o-', linewidth=2, markersize=8)
        ax1.axhline(y=np.mean(cv_scores), color='r', linestyle='--', alpha=0.7, label=f'Mean: {np.mean(cv_scores):.4f}')
        ax1.fill_between(x_vals, 
                        np.mean(cv_scores) - np.std(cv_scores), 
                        np.mean(cv_scores) + np.std(cv_scores), 
                        alpha=0.2, color='red')
        ax1.set_title(f"{title} - Scores Over Time")
        ax1.set_xlabel("Fold" if cv_dates is None else "Date")
        ax1.set_ylabel("Score")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot distribution
        ax2.hist(cv_scores, bins=min(10, len(cv_scores)), alpha=0.7, edgecolor='black')
        ax2.axvline(x=np.mean(cv_scores), color='r', linestyle='--', alpha=0.7, label=f'Mean: {np.mean(cv_scores):.4f}')
        ax2.set_title(f"{title} - Distribution")
        ax2.set_xlabel("Score")
        ax2.set_ylabel("Frequency")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        metrics = calculate_cv_metrics(cv_scores)
        print("\nCross-Validation Metrics:")
        print("-" * 40)
        for key, value in metrics.items():
            print(f"{key:15s}: {value:.4f}")
            
    except ImportError:
        print("Matplotlib not available. Showing numeric results only.")
        metrics = calculate_cv_metrics(cv_scores)
        print("\nCross-Validation Metrics:")
        print("-" * 40)
        for key, value in metrics.items():
            print(f"{key:15s}: {value:.4f}")

# Example usage and utility functions
def demonstrate_purged_cv():
    """Demonstrate the purged CV with sample data"""
    print("Demonstrating Purged Walk-Forward Cross-Validation")
    print("=" * 50)
    
    # Create sample time series data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Generate time series with some autocorrelation
    X = np.random.randn(n_samples, n_features)
    # Add some temporal dependencies
    for i in range(1, n_samples):
        X[i] = 0.7 * X[i-1] + 0.3 * X[i]
    
    # Create target with some temporal pattern
    y = np.zeros(n_samples)
    for i in range(10, n_samples):
        y[i] = np.sign(np.sum(X[i-10:i, 0])) + 0.1 * np.random.randn()
    y = (y > 0).astype(int)
    
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    y_series = pd.Series(y)
    
    # Create purged CV
    cv = PurgedWalkForwardCV(
        n_splits=5,
        test_size=100,
        purge_size=10,
        embargo_size=5
    )
    
    print(f"Total samples: {n_samples}")
    print(f"CV splits: {cv.n_splits}")
    print(f"Test size: {cv.test_size}")
    print(f"Purge size: {cv.purge_size}")
    print(f"Embargo size: {cv.embargo_size}")
    print()
    
    # Show split information
    print("Split Information:")
    print("-" * 20)
    for i, (train_idx, test_idx) in enumerate(cv.split(X_df)):
        print(f"Fold {i+1}: Train [{train_idx[0]:4d}:{train_idx[-1]:4d}] "
              f"Test [{test_idx[0]:4d}:{test_idx[-1]:4d}] "
              f"(Train size: {len(train_idx)}, Test size: {len(test_idx)})")
    
    return X_df, y_series, cv

if __name__ == "__main__":
    # Demonstrate the purged CV
    X, y, cv = demonstrate_purged_cv()
    
    # Example with a simple model
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        
        model = LogisticRegression(random_state=42)
        scores = purged_cv_score(model, X, y, cv=cv, scoring='accuracy', verbose=1)
        
        print(f"\nPurged CV Scores: {scores}")
        print(f"Mean CV Score: {np.mean(scores):.4f} (+/- {np.std(scores) * 2:.4f})")
        
        # Plot results
        plot_cv_results(scores, title="Purged Walk-Forward CV")
        
    except ImportError:
        print("Scikit-learn not available for demonstration")