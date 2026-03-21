"""Parse stored rounds into training DataFrames.

Primary label source: analysis/seed_k.json ground_truth
(only rounds where analysis exists are included).

Returns:
    cell_df  — one row per (round_id, seed_idx, row, col) with features + target_class
    round_df — one row per (round_id, seed_idx) with round-level aggregates
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..map.encoding import encode_grid
from ..map.features import build_feature_matrix, FEATURE_NAMES
from ..utils.logging import get_logger

log = get_logger(__name__)


def build_history(
    data_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scan all stored completed rounds and build training DataFrames.

    Args:
        data_dir: root data directory (contains rounds/)

    Returns:
        (cell_df, round_df)
    """
    rounds_dir = Path(data_dir) / "rounds"
    if not rounds_dir.exists():
        return pd.DataFrame(), pd.DataFrame()

    cell_rows: list[dict] = []
    round_rows: list[dict] = []

    for round_dir in sorted(rounds_dir.iterdir()):
        if not round_dir.is_dir():
            continue
        round_id = round_dir.name

        analysis_dir = round_dir / "analysis"
        if not analysis_dir.exists():
            continue

        analysis_files = list(analysis_dir.glob("seed_*.json"))
        if not analysis_files:
            continue

        detail_path = round_dir / "detail.json"
        if not detail_path.exists():
            continue

        try:
            with open(detail_path) as f:
                detail = json.load(f)
        except Exception as e:
            log.warning(f"Could not load {detail_path}: {e}")
            continue

        H = detail["map_height"]
        W = detail["map_width"]

        for seed_path in sorted(analysis_files):
            seed_idx = int(seed_path.stem.split("_")[1])
            try:
                with open(seed_path) as f:
                    analysis = json.load(f)
            except Exception as e:
                log.warning(f"Could not load {seed_path}: {e}")
                continue

            ground_truth = analysis.get("ground_truth")
            if ground_truth is None:
                log.warning(f"No ground_truth in {seed_path}")
                continue

            # ground_truth: H x W x 6 probability distribution or H x W int grid
            gt = np.array(ground_truth)
            if gt.ndim == 2:
                # Integer label grid → one-hot with light smoothing
                gt_classes = gt.astype(np.int32)
                gt_probs = np.eye(6, dtype=np.float32)[gt_classes]  # [H, W, 6]
                # Light smoothing
                gt_probs = gt_probs * 0.95 + 0.005
                gt_probs /= gt_probs.sum(axis=-1, keepdims=True)
            else:
                gt_probs = gt.astype(np.float32)

            # Load raw grid for this seed
            raw_path = round_dir / f"raw_grids_seed_{seed_idx}.npz"
            if not raw_path.exists():
                # Try from detail
                raw = np.array(detail["initial_states"][seed_idx]["grid"], dtype=np.int16)
            else:
                raw = np.load(raw_path)["grid"].astype(np.int16)

            # Build round features from query observations
            rf = _build_round_features_from_queries(round_dir, seed_idx)

            # Build feature matrix [H*W, n_feats]
            try:
                feat_matrix = build_feature_matrix(raw, round_features=rf)
            except Exception as e:
                log.warning(f"Feature build failed for {round_id} seed {seed_idx}: {e}")
                continue

            # Target class = argmax of ground truth distribution
            target_class = np.argmax(gt_probs, axis=-1)  # [H, W]

            for row in range(H):
                for col in range(W):
                    feat_idx = row * W + col
                    r: dict = {
                        "round_id": round_id,
                        "seed_idx": seed_idx,
                        "row": row,
                        "col": col,
                        "target_class": int(target_class[row, col]),
                    }
                    for fi, fn in enumerate(FEATURE_NAMES):
                        r[fn] = float(feat_matrix[feat_idx, fi])
                    # Soft label columns
                    for ci in range(6):
                        r[f"gt_prob_{ci}"] = float(gt_probs[row, col, ci])
                    cell_rows.append(r)

            score = analysis.get("score")
            round_rows.append({
                "round_id": round_id,
                "seed_idx": seed_idx,
                "score": score,
                "H": H, "W": W,
            })
            log.info(f"Ingested {round_id} seed {seed_idx} ({H}x{W})")

    cell_df = pd.DataFrame(cell_rows)
    round_df = pd.DataFrame(round_rows)
    return cell_df, round_df


def _build_round_features_from_queries(
    round_dir: Path, seed_idx: int
) -> dict:
    """Estimate round-level features from stored query files."""
    queries_dir = round_dir / "queries"
    if not queries_dir.exists():
        return {}

    ruin_count = settlement_count = port_count = total = 0
    for qf in queries_dir.glob("q_*.json"):
        try:
            with open(qf) as f:
                q = json.load(f)
            if q.get("metadata", {}).get("seed_idx") != seed_idx:
                continue
            grid = q.get("result", {}).get("grid", [])
            for row in grid:
                for code in row:
                    total += 1
                    if code == 3:
                        ruin_count += 1
                    elif code == 1:
                        settlement_count += 1
                    elif code == 2:
                        port_count += 1
        except Exception:
            continue

    if total == 0:
        return {}
    sett_total = settlement_count + ruin_count
    return {
        "ruin_rate": ruin_count / total,
        "port_rate": port_count / total,
        "settlement_survival": settlement_count / max(sett_total, 1),
    }
