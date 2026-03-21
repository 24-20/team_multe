"""Post-hoc calibration for the LightGBM model.

Two methods:
  - temperature_scale (default): single scalar T on log-probabilities.
    Works well with few validation rounds (as low as 2-3 rounds).
  - isotonic: per-class isotonic regression. Needs more data; kept for
    when 5+ completed rounds are available.
"""
from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..utils.logging import get_logger

log = get_logger(__name__)


class TemperatureScaledModel:
    """Thin wrapper that applies temperature scaling to predict_proba output.

    Temperature scaling: p_cal = softmax(log(p_raw) / T)
    T > 1 → softer (more uncertain); T < 1 → sharper.
    """

    def __init__(self, base_model, temperature: float) -> None:
        self.base_model = base_model
        self.temperature = temperature
        # Proxy class attributes so downstream code can read model.classes_
        if hasattr(base_model, "classes_"):
            self.classes_ = base_model.classes_

    def predict_proba(self, X) -> np.ndarray:
        p = self.base_model.predict_proba(X)
        if abs(self.temperature - 1.0) < 1e-6:
            return p
        log_p = np.log(np.clip(p, 1e-10, 1.0))
        scaled = log_p / self.temperature
        # Numerically stable softmax
        scaled -= scaled.max(axis=1, keepdims=True)
        exp_s = np.exp(scaled)
        return exp_s / exp_s.sum(axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


def _find_temperature(p_raw: np.ndarray, y_true: np.ndarray) -> float:
    """Find T in [0.1, 10] that minimises NLL on (p_raw, y_true).

    Args:
        p_raw:  [N, 6] predicted probabilities from uncalibrated model
        y_true: [N] integer class labels
    Returns:
        optimal temperature (float)
    """
    from scipy.optimize import minimize_scalar

    log_p = np.log(np.clip(p_raw, 1e-10, 1.0))

    def nll(T: float) -> float:
        scaled = log_p / T
        scaled -= scaled.max(axis=1, keepdims=True)
        exp_s = np.exp(scaled)
        probs = exp_s / exp_s.sum(axis=1, keepdims=True)
        return -float(np.mean(np.log(probs[np.arange(len(y_true)), y_true] + 1e-10)))

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return float(result.x)


def calibrate_model(
    model,
    val_df,
    models_dir: str | Path,
    method: str = "temperature",
) -> None:
    """Calibrate the LightGBM model and save calibrated version.

    Args:
        model:      fitted LGBMClassifier (or already-wrapped model)
        val_df:     validation DataFrame with 'target_class' + feature columns
        models_dir: directory containing lgbm.pkl + lgbm_metadata.json
        method:     "temperature" (default) or "isotonic"
    """
    from ..map.features import FEATURE_NAMES

    models_dir = Path(models_dir)
    meta_path = models_dir / "lgbm_metadata.json"

    feat_cols = [c for c in FEATURE_NAMES if c in val_df.columns]
    X_val = val_df[feat_cols].values.astype(np.float32)
    y_val = val_df["target_class"].values.astype(np.int32)

    if method == "temperature":
        # Get raw (uncalibrated) probabilities
        # If model is already a TemperatureScaledModel, unwrap first
        base = model.base_model if isinstance(model, TemperatureScaledModel) else model
        p_raw = base.predict_proba(X_val)
        T = _find_temperature(p_raw, y_val)
        log.info(f"Temperature scaling: T={T:.4f}")
        calibrated = TemperatureScaledModel(base, T)
        cal_version = f"temperature_T{T:.4f}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"

    elif method == "isotonic":
        try:
            from sklearn.calibration import CalibratedClassifierCV
        except ImportError:
            raise ImportError("scikit-learn is required for isotonic calibration")
        # Unwrap if needed
        base = model.base_model if isinstance(model, TemperatureScaledModel) else model
        calibrated = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
        calibrated.fit(X_val, y_val)
        cal_version = f"isotonic_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"

    else:
        raise ValueError(f"Unknown calibration method: {method!r}")

    cal_path = models_dir / "lgbm_calibrated.pkl"
    with open(cal_path, "wb") as f:
        pickle.dump(calibrated, f)

    # Update metadata
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {}

    meta["calibration_version"] = cal_version
    meta["calibrated_model_path"] = str(cal_path)
    if method == "temperature":
        meta["temperature"] = float(
            calibrated.temperature if isinstance(calibrated, TemperatureScaledModel)
            else 1.0
        )
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log.info(f"Calibrated model saved to {cal_path} (version: {cal_version})")
