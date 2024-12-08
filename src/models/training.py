"""
Training utilities for the CryptoTransformer model.
Includes cross-validation and hyperparameter optimization.
"""
import tensorflow as tf
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from typing import Tuple, List, Dict, Any
from .transformer_model import create_model

class ModelTrainer:
    def __init__(
        self,
        sequence_length: int,
        num_features: int,
        n_splits: int = 5,
        val_size: float = 0.2
    ):
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.n_splits = n_splits
        self.val_size = val_size
        self.best_model = None
        self.best_params = None
        self.best_score = float('inf')

    def prepare_direction_labels(self, y: np.ndarray) -> np.ndarray:
        """Prepare binary labels for direction prediction."""
        return (np.diff(y, prepend=y[0]) > 0).astype(np.float32)

    def train_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_params: Dict[str, Any],
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 10,
        verbose: int = 1
    ) -> Tuple[tf.keras.Model, Dict[str, float]]:
        """
        Train a single model with early stopping and learning rate reduction.
        """
        # Prepare direction labels
        y_direction = self.prepare_direction_labels(y)

        # Create model
        model = create_model(
            sequence_length=self.sequence_length,
            num_features=self.num_features,
            **model_params
        )

        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_price_output_mae',
                patience=patience,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_price_output_mae',
                factor=0.5,
                patience=patience // 2,
                min_lr=1e-6
            )
        ]

        # Train model
        history = model.fit(
            X,
            {
                'price_output': y,
                'direction_output': y_direction
            },
            validation_split=self.val_size,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )

        # Get best validation scores
        val_scores = {
            'val_price_mae': min(history.history['val_price_output_mae']),
            'val_price_mse': min(history.history['val_price_output_mse']),
            'val_direction_accuracy': max(history.history['val_direction_output_accuracy'])
        }

        return model, val_scores

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_params: Dict[str, Any],
        **train_params
    ) -> Dict[str, List[float]]:
        """
        Perform time series cross-validation.
        """
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        scores = {
            'price_mae': [],
            'price_mse': [],
            'direction_accuracy': []
        }

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            y_val_direction = self.prepare_direction_labels(y_val)

            # Train model
            model = create_model(
                sequence_length=self.sequence_length,
                num_features=self.num_features,
                **model_params
            )

            model.fit(
                X_train,
                {
                    'price_output': y_train,
                    'direction_output': self.prepare_direction_labels(y_train)
                },
                **train_params
            )

            # Evaluate
            val_results = model.evaluate(
                X_val,
                {
                    'price_output': y_val,
                    'direction_output': y_val_direction
                },
                verbose=0
            )

            scores['price_mae'].append(val_results[3])  # MAE for price prediction
            scores['price_mse'].append(val_results[4])  # MSE for price prediction
            scores['direction_accuracy'].append(val_results[5])  # Accuracy for direction

        return scores

    def optimize_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: Dict[str, List[Any]],
        **train_params
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Perform grid search for hyperparameter optimization.
        """
        from itertools import product

        # Generate all combinations of parameters
        keys = param_grid.keys()
        values = param_grid.values()
        param_combinations = [dict(zip(keys, v)) for v in product(*values)]

        best_score = float('inf')
        best_params = None
        best_metrics = None

        for params in param_combinations:
            # Perform cross-validation
            cv_scores = self.cross_validate(X, y, params, **train_params)

            # Calculate average MAE (primary metric)
            avg_mae = np.mean(cv_scores['price_mae'])

            if avg_mae < best_score:
                best_score = avg_mae
                best_params = params
                best_metrics = {
                    'mae': np.mean(cv_scores['price_mae']),
                    'mse': np.mean(cv_scores['price_mse']),
                    'direction_accuracy': np.mean(cv_scores['direction_accuracy'])
                }

        # Train final model with best parameters
        final_model, _ = self.train_model(X, y, best_params, **train_params)

        self.best_model = final_model
        self.best_params = best_params
        self.best_score = best_score

        return best_params, best_metrics

    def save_model(self, path: str):
        """Save the best model to disk."""
        if self.best_model is not None:
            self.best_model.save(path)
