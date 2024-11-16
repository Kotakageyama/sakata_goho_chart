"""
Script to run model evaluation and generate performance reports.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from src.data.data_loader import CryptoDataLoader
from src.models.transformer_model import CryptoTransformer, create_model
from src.models.strategy import TransformerStrategy
from src.evaluation.model_evaluation import ModelEvaluator

def prepare_sequences(data: pd.DataFrame, sequence_length: int = 60) -> tuple:
    """
    Prepare sequences for the transformer model.
    """
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    sequences = []
    targets = []

    for i in range(len(data) - sequence_length):
        seq = data[features].iloc[i:i+sequence_length].values
        target = data['Close'].iloc[i+sequence_length]
        sequences.append(seq)
        targets.append(target)

    return np.array(sequences), np.array(targets)

def main():
    # Load data
    data_path = os.path.join(project_root, "data", "ETHUSD_2Year_2022-11-15_2024-11-15.csv")
    loader = CryptoDataLoader()
    data = loader.load_data(data_path)

    # Prepare sequences
    sequence_length = 60
    num_features = 5  # OHLCV
    X, y = prepare_sequences(data, sequence_length)

    # Initialize evaluator
    evaluator = ModelEvaluator(data, n_splits=5)

    # Model parameters
    base_params = {
        'sequence_length': sequence_length,
        'num_features': num_features,
        'd_model': 64,
        'num_heads': 8,
        'ff_dim': 128,
        'num_transformer_blocks': 4,
        'mlp_units': [64, 32],
        'dropout': 0.1,
        'mlp_dropout': 0.2
    }

    print("最適化前のモデル評価開始...")

    # Initialize models
    old_model = create_model(**base_params)
    new_model = create_model(**base_params)

    # Run initial cross-validation
    initial_metrics = evaluator.cross_validate(new_model, TransformerStrategy)
    print("\n初期クロスバリデーション結果:")
    for metric, values in initial_metrics.items():
        print(f"{metric}: {np.mean(values):.4f} (±{np.std(values):.4f})")

    print("\nハイパーパラメータ最適化開始...")

    # Optimize hyperparameters
    best_params = evaluator.optimize_hyperparameters(n_trials=50)
    print("\n最適化されたパラメータ:")
    for param, value in best_params.items():
        print(f"{param}: {value}")

    # Create optimized model
    optimized_params = base_params.copy()
    optimized_params.update(best_params)
    optimized_model = create_model(**optimized_params)

    print("\nモデル比較開始...")

    # Compare models
    comparison = evaluator.compare_models(old_model, optimized_model)
    print("\nモデル比較結果:")
    for model_name, metrics in comparison.items():
        print(f"\n{model_name}モデル:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    # Generate and save report
    report = evaluator.generate_report()
    report_path = os.path.join(project_root, "reports", "model_evaluation_report.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n評価レポートを保存しました: {report_path}")

if __name__ == "__main__":
    main()
