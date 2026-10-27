"""
scripts/train_model.py
Trains the Isolation Forest model for anomaly detection on invoice financials.
Saves the serialized model artifact to artifacts/isolation_forest_tuned.pkl.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

FEATURES = [
    "subtotal",
    "cgst_amount",
    "sgst_amount",
    "igst_amount",
    "tax_amount",
    "discount_amount",
    "total_amount",
    "item_count",
    "quantity",
]


def generate_synthetic_training_data(n_samples: int = 2000) -> pd.DataFrame:
    """Generate realistic synthetic Indian B2B invoice data for initial model bootstrap."""
    np.random.seed(42)
    subtotals = np.random.exponential(scale=25000, size=n_samples) + 500
    subtotals = np.round(subtotals, 2)

    # 70% intra-state (CGST+SGST), 30% inter-state (IGST)
    is_interstate = np.random.rand(n_samples) < 0.3
    tax_rate = 0.18

    cgst = np.where(~is_interstate, np.round(subtotals * (tax_rate / 2), 2), 0.0)
    sgst = np.where(~is_interstate, np.round(subtotals * (tax_rate / 2), 2), 0.0)
    igst = np.where(is_interstate, np.round(subtotals * tax_rate, 2), 0.0)
    taxes = cgst + sgst + igst

    discounts = np.where(
        np.random.rand(n_samples) < 0.25,
        np.round(subtotals * np.random.uniform(0.02, 0.10, size=n_samples), 2),
        0.0,
    )
    totals = np.round(subtotals + taxes - discounts, 2)

    item_counts = np.random.randint(1, 15, size=n_samples)
    quantities = item_counts * np.random.randint(1, 10, size=n_samples)

    return pd.DataFrame({
        "subtotal": subtotals,
        "cgst_amount": cgst,
        "sgst_amount": sgst,
        "igst_amount": igst,
        "tax_amount": taxes,
        "discount_amount": discounts,
        "total_amount": totals,
        "item_count": item_counts,
        "quantity": quantities,
    })


def train(
    data_path: str | None = None,
    output_path: str = "artifacts/isolation_forest_tuned.pkl",
    contamination: float = 0.05,
    random_state: int = 42,
) -> Path:
    """Train Isolation Forest on invoice features and save to disk."""
    if data_path and Path(data_path).exists():
        print(f"Loading training data from {data_path}...")
        df = pd.read_csv(data_path)
    else:
        print("No CSV provided or file not found. Generating realistic training data...")
        df = generate_synthetic_training_data()

    # Ensure all required features are present
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Training dataset is missing required features: {missing}")

    from sklearn.impute import SimpleImputer
    X = df[FEATURES]
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=FEATURES)

    print(f"Training IsolationForest on {len(X)} records across {len(FEATURES)} features...")
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_imputed)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_file)
    print(f"Model successfully saved to {out_file.resolve()}")

    imputer_file = out_file.parent / "isolation_forest_imputer.pkl"
    joblib.dump(imputer, imputer_file)
    print(f"Imputer successfully saved to {imputer_file.resolve()}")

    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Isolation Forest for FinDoc AI")
    parser.add_argument("--data", type=str, default=None, help="Path to training CSV")
    parser.add_argument("--output", type=str, default="artifacts/isolation_forest_tuned.pkl", help="Output pkl path")
    parser.add_argument("--contamination", type=float, default=0.05, help="Anomaly rate (default: 0.05)")
    args = parser.parse_args()

    train(args.data, args.output, args.contamination)
