"""
train.py
Reproducible training script for OPD Triage and Department Routing models.
Usage:
    python train.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.ml_model import train_models

def main():
    print("=" * 60)
    print("AI-Assisted OPD Triage - Model Training Pipeline")
    print("=" * 60)
    t0 = time.time()

    dataset_path = os.path.join("data", "processed", "dev_dataset.csv")
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found. Running Phase 1 data preparation...")
        import subprocess
        subprocess.run([sys.executable, "scripts/phase1_data_quality.py"], check=True)

    results = train_models(dataset_path=dataset_path, save_models=True)
    t1 = time.time()

    print("=" * 60)
    print(f"Training completed successfully in {t1 - t0:.1f} seconds.")
    print("Validation Metrics:")
    for k, v in results["val_metrics"].items():
        print(f"  {k}: {v:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
