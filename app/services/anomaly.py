"""
app/services/anomaly.py
Loads the saved Isolation Forest model and runs inference.
Refactored from untitled1.py / untitled2.py (Colab).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Feature list — must exactly match the features the model was trained on.
# ---------------------------------------------------------------------------

MODEL_FEATURES = [
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

# Default artifact paths (can be overridden via env vars)
_ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
DEFAULT_MODEL_PATH = Path(
    os.getenv("IF_MODEL_PATH", str(_ARTIFACTS_DIR / "isolation_forest_tuned.pkl"))
)
DEFAULT_IMPUTER_PATH = Path(
    os.getenv("IF_IMPUTER_PATH", str(_ARTIFACTS_DIR / "isolation_forest_imputer.pkl"))
)


class AnomalyDetector:
    """
    Thin wrapper around the saved Isolation Forest model.
    Handles loading, feature preparation, imputation, and scoring.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        imputer_path: str | Path = DEFAULT_IMPUTER_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.imputer_path = Path(imputer_path)
        self._model = None
        self._imputer = None
        self._loaded = False

    def load(self) -> None:
        """Load model and imputer from disk. Call once at application startup."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self._model = joblib.load(self.model_path)

        if self.imputer_path.exists():
            self._imputer = joblib.load(self.imputer_path)
        else:
            from sklearn.impute import SimpleImputer
            self._imputer = SimpleImputer(strategy="constant", fill_value=0.0)
            # Fit on dummy zero data to initialize shape
            self._imputer.fit(np.zeros((1, len(MODEL_FEATURES))))

        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(
        self, fields: Dict[str, Any]
    ) -> Tuple[bool, Optional[float], bool]:
        """
        Run anomaly detection on a single invoice dict.

        Returns:
            (isolation_anomaly, isolation_score, model_input_invalid)
            - isolation_anomaly: True when the model flags an anomaly
            - isolation_score:   Raw decision function score (negative = more anomalous)
            - model_input_invalid: True when required features are missing
        """
        if not self._loaded:
            self.load()

        # Build a 1-row feature DataFrame
        X = pd.DataFrame([{f: fields.get(f) for f in MODEL_FEATURES}])
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.apply(pd.to_numeric, errors="coerce")

        if X.isna().any(axis=1).iloc[0]:
            return False, None, True  # cannot score — missing inputs

        X_imputed = pd.DataFrame(
            self._imputer.transform(X), columns=MODEL_FEATURES
        )

        prediction = self._model.predict(X_imputed)[0]         # -1 or 1
        score = float(self._model.decision_function(X_imputed)[0])  # raw score

        return (prediction == -1), score, False

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run anomaly detection on a DataFrame of invoices.

        Adds columns: isolation_anomaly, isolation_score, model_input_invalid.
        """
        if not self._loaded:
            self.load()

        result = df.copy()
        result["isolation_anomaly"] = False
        result["isolation_score"] = None
        result["model_input_invalid"] = False

        X = result.reindex(columns=MODEL_FEATURES).replace(
            [np.inf, -np.inf], np.nan
        ).apply(pd.to_numeric, errors="coerce")

        invalid = X.isna().any(axis=1)
        result["model_input_invalid"] = invalid

        valid = ~invalid
        if valid.any():
            X_valid = pd.DataFrame(
                self._imputer.transform(X.loc[valid]), columns=MODEL_FEATURES
            )
            preds = self._model.predict(X_valid)
            scores = self._model.decision_function(X_valid)

            result.loc[valid, "isolation_anomaly"] = preds == -1
            result.loc[valid, "isolation_score"] = scores

        return result


# Module-level singleton — loaded lazily on first use.
_detector: Optional[AnomalyDetector] = None


def get_detector() -> AnomalyDetector:
    """Return the shared AnomalyDetector instance, loading it if needed."""
    global _detector
    if _detector is None:
        _detector = AnomalyDetector()
        _detector.load()
    return _detector
