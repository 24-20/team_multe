"""Replay a stored round locally for debugging and analysis.

Loads detail.json + all stored queries in order, reconstructs RoundState
step by step, then runs inference and reports stats.

Does NOT call any live API.

Usage:
    python -m astar_island.scripts.replay_round --round_id ID [--model lgbm|baseline]
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

from dotenv import load_dotenv
import numpy as np

_pkg_root = Path(__file__).resolve().parents[2]
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

load_dotenv()

from astar_island.online.round_state import RoundState
from astar_island.online.infer import infer_full_map
from astar_island.storage.store import RoundStore
from astar_island.utils.io import save_npz
from astar_island.utils.logging import get_logger

log = get_logger("replay_round")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay stored round offline")
    parser.add_argument("--round_id", required=True, help="Round ID to replay")
    parser.add_argument(
        "--model", choices=["lgbm", "baseline"], default="baseline",
        help="Which model to use for inference",
    )
    parser.add_argument("--save-tensors", action="store_true",
                        help="Save replayed prediction tensors to disk")
    args = parser.parse_args()

    data_dir = Path(os.environ.get("DATA_DIR", "astar_island/data"))
    if not data_dir.is_absolute():
        data_dir = Path.cwd() / data_dir

    store = RoundStore(data_dir, args.round_id)

    # ── Load detail ────────────────────────────────────────────────────
    log.info(f"Replaying round {args.round_id}...")
    detail = store.load_detail()
    state = RoundState.from_detail(detail)
    log.info(
        f"Map: {state.width}x{state.height}, {state.seeds_count} seeds"
    )

    # ── Replay queries in order ────────────────────────────────────────
    queries = store.load_queries()
    log.info(f"Replaying {len(queries)} stored queries...")

    for q in queries:
        meta = q.get("metadata", {})
        result = q.get("result", {})
        seed_idx = meta.get("seed_idx", meta.get("seed_index", 0))
        vp = result.get("viewport", {})
        x = vp.get("x", meta.get("x", 0))
        y = vp.get("y", meta.get("y", 0))
        w = vp.get("w", meta.get("w", 15))
        h = vp.get("h", meta.get("h", 15))
        grid_result = result.get("grid", [])
        settlements = result.get("settlements")

        if not grid_result:
            continue

        state.update_from_query(
            seed_idx=seed_idx,
            viewport_x=x, viewport_y=y,
            viewport_w=w, viewport_h=h,
            grid_result=grid_result,
            settlements_result=settlements,
        )

    log.info("All queries replayed")
    for s in range(state.seeds_count):
        log.info(f"Seed {s}: coverage={state.coverage_fraction(s):.1%}")

    # ── Load model ─────────────────────────────────────────────────────
    lgbm_model = None
    if args.model == "lgbm":
        models_dir = data_dir.parent / "models"
        if not models_dir.is_absolute():
            models_dir = Path.cwd() / "astar_island" / "models"
        model_path = models_dir / "lgbm.pkl"
        if model_path.exists():
            with open(model_path, "rb") as f:
                lgbm_model = pickle.load(f)
            log.info("Loaded LightGBM model")
        else:
            log.warning(f"LightGBM model not found at {model_path} — using baseline")

    # ── Infer ──────────────────────────────────────────────────────────
    pred = infer_full_map(state, lgbm_model=lgbm_model)
    log.info(f"Prediction shape: {pred.shape}")

    for s in range(state.seeds_count):
        t = pred[s]
        log.info(
            f"Seed {s}: min={t.min():.4f} max={t.max():.4f} "
            f"sum_range=[{t.sum(-1).min():.4f}, {t.sum(-1).max():.4f}]"
        )

    if args.save_tensors:
        out_dir = store.root / "replayed_predictions"
        out_dir.mkdir(exist_ok=True)
        for s in range(state.seeds_count):
            out_path = out_dir / f"seed_{s}_tensor.npz"
            save_npz(out_path, tensor=pred[s])
            log.info(f"Saved replayed prediction to {out_path}")


if __name__ == "__main__":
    main()
