"""Train LightGBM classifier with GroupKFold cross-validation by round_id."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from ..map.features import FEATURE_NAMES
from ..utils.logging import get_logger

log = get_logger(__name__)


def train_lgbm(
    df: pd.DataFrame,
    models_dir: str | Path,
    n_splits: int = 5,
) -> tuple:
    """Train LightGBM multiclass classifier and save to disk.

    Args:
        df:          cell DataFrame with FEATURE_NAMES columns + target_class + round_id
        models_dir:  directory to save model + metadata
        n_splits:    GroupKFold splits

    Returns:
        (model, metadata_dict)
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("lightgbm is required: pip install lightgbm")

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    feat_cols = [c for c in FEATURE_NAMES if c in df.columns]
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        log.warning(f"Missing feature columns (will be ignored): {missing}")

    X = df[feat_cols].values.astype(np.float32)
    y = df["target_class"].values.astype(np.int32)
    groups = df["round_id"].values

    log.info(
        f"Training LightGBM on {len(X)} samples, "
        f"{len(feat_cols)} features, "
        f"{len(np.unique(groups))} rounds"
    )

    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    oof_preds = np.zeros((len(y), 6), dtype=np.float32)

    model = lgb.LGBMClassifier(
        num_leaves=63,
        n_estimators=500,
        learning_rate=0.05,
        class_weight="balanced",
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        n_jobs=-1,
        verbose=-1,
    )

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        fold_model = lgb.LGBMClassifier(
            num_leaves=63, n_estimators=500, learning_rate=0.05,
            class_weight="balanced", subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, n_jobs=-1, verbose=-1,
        )
        fold_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        p_val = fold_model.predict_proba(X_val)
        # Map model classes_ back to 6-class columns (some classes may be absent)
        for ci, cls in enumerate(fold_model.classes_):
            oof_preds[val_idx, int(cls)] = p_val[:, ci]
        log.info(f"Fold {fold+1}: val size={len(val_idx)}, classes={list(fold_model.classes_)}")

    # Compute OOF KL divergence (vs one-hot)
    from ..utils.math import kl_div
    oof_kl_values = []
    for i in range(len(y)):
        gt = np.zeros(6)
        gt[y[i]] = 1.0
        kl = kl_div(gt, np.maximum(oof_preds[i], 1e-7))
        oof_kl_values.append(kl)
    val_kl = float(np.mean([v for v in oof_kl_values if np.isfinite(v)]))
    log.info(f"OOF KL divergence: {val_kl:.4f}")

    # Train final model on all data
    model.fit(X, y)

    # Auto-calibrate using OOF predictions (held-out, unbiased)
    from .calibrate import _find_temperature, TemperatureScaledModel
    oof_finite = np.isfinite(oof_preds).all(axis=1)
    T = _find_temperature(oof_preds[oof_finite], y[oof_finite])
    calibrated_model = TemperatureScaledModel(model, T)
    log.info(f"Temperature scaling (OOF): T={T:.4f}")

    # Save both raw and calibrated models
    model_path = models_dir / "lgbm.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    cal_path = models_dir / "lgbm_calibrated.pkl"
    with open(cal_path, "wb") as f:
        pickle.dump(calibrated_model, f)

    # Git commit hash (best effort)
    git_hash = _get_git_hash()

    metadata = {
        "feature_columns": feat_cols,
        "training_round_ids": sorted(np.unique(groups).tolist()),
        "n_samples": len(X),
        "n_features": len(feat_cols),
        "val_score_kl": val_kl,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_hash,
        "calibration_version": f"temperature_T{T:.4f}_oof",
        "calibrated_model_path": str(cal_path),
        "temperature": T,
        "model_path": str(model_path),
    }
    meta_path = models_dir / "lgbm_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    log.info(f"Model saved to {model_path}, calibrated to {cal_path}, KL={val_kl:.4f}")
    return calibrated_model, metadata


def _get_git_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() or None
    except Exception:
        return None
