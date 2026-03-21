"""Main entry point for running a live Astar Island round.

Usage:
    python -m astar_island.scripts.run_online_round [--mode live|dryrun] [--round_id ID]

Control flow (submit-early strategy):
    1. Load env + config; init client
    2. get_active_round() (or --round_id)
    3. get_round_detail() → RoundState
    4. Init RoundStore; persist meta + detail + initial grids
    5. [SUBMIT 1] Infer baseline (no queries yet) → submit all seeds
    6. Phase 1: coverage queries (8 per seed) → update state
    7. [SUBMIT 2] Re-infer + resubmit
    8. Phase 2: adaptive queries on remaining budget → update state
    9. [SUBMIT 3] Final re-infer + resubmit
    10. Log summary
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

from dotenv import load_dotenv

# Ensure package root is importable when run as a script
_pkg_root = Path(__file__).resolve().parents[2]
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

load_dotenv()

from astar_island.api.client import AstarClient
from astar_island.online.round_state import RoundState
from astar_island.online.query_policy import QueryPolicy
from astar_island.online.infer import infer_full_map
from astar_island.online.submit import submit_all_seeds
from astar_island.storage.store import RoundStore
from astar_island.validators import validate_round_detail
from astar_island.utils.logging import get_logger

log = get_logger("run_online_round")


def _load_models(models_dir: Path) -> tuple:
    """Load best available models. Returns (cnn_model, lgbm_model).

    Priority:
        - CNN (cnn.pt) if available → used instead of LightGBM
        - LightGBM (lgbm_calibrated.pkl / lgbm.pkl) as fallback
        - Both can be None → baseline posterior only

    To force LightGBM: delete or rename astar_island/models/cnn.pt
    """
    cnn_model = None
    try:
        from astar_island.training.train_cnn import load_cnn
        cnn_model = load_cnn(models_dir)
    except Exception as e:
        log.warning(f"Could not load CNN: {e}")

    lgbm_model = None
    if cnn_model is None:
        # Only load LightGBM if CNN unavailable
        for path, label in [
            (models_dir / "lgbm_calibrated.pkl", "calibrated"),
            (models_dir / "lgbm.pkl", "raw"),
        ]:
            if path.exists():
                try:
                    with open(path, "rb") as f:
                        lgbm_model = pickle.load(f)
                    log.info(f"Loaded LightGBM ({label}) from {path}")
                    break
                except Exception as e:
                    log.warning(f"Failed to load LightGBM {label}: {e}")

    if cnn_model is None and lgbm_model is None:
        log.info("No ML model found — using baseline posterior only")

    return cnn_model, lgbm_model


def _execute_queries(
    client: AstarClient,
    state: RoundState,
    store: RoundStore,
    specs,
    dry_run: bool,
) -> None:
    """Execute a list of QuerySpec objects, update state, persist results."""
    for spec in specs:
        if dry_run:
            log.info(
                f"[DRYRUN] simulate seed={spec.seed_idx} "
                f"x={spec.x} y={spec.y} w={spec.w} h={spec.h}"
            )
            continue

        try:
            result = client.simulate(
                round_id=state.round_id,
                seed_index=spec.seed_idx,
                x=spec.x, y=spec.y,
                w=spec.w, h=spec.h,
            )
        except Exception as e:
            log.error(f"simulate() failed: {e}")
            continue

        query_idx = store.query_count()
        store.save_query(
            query_idx=query_idx,
            result=result,
            metadata={
                "seed_idx": spec.seed_idx,
                "x": spec.x, "y": spec.y,
                "w": spec.w, "h": spec.h,
                "phase": spec._asdict().get("phase", "coverage"),
            },
        )

        state.update_from_query(
            seed_idx=spec.seed_idx,
            viewport_x=spec.x,
            viewport_y=spec.y,
            viewport_w=spec.w,
            viewport_h=spec.h,
            grid_result=result["grid"],
            settlements_result=result.get("settlements"),
            query_metadata={"x": spec.x, "y": spec.y},
        )

        log.info(
            f"Query {query_idx}: seed={spec.seed_idx} ({spec.x},{spec.y}) "
            f"coverage={state.coverage_fraction(spec.seed_idx):.1%}"
        )


def run(args: argparse.Namespace) -> None:
    dry_run = args.mode == "dryrun"

    # ── Setup ──────────────────────────────────────────────────────────
    data_dir = Path(os.environ.get("DATA_DIR", "astar_island/data"))
    models_dir = data_dir.parent / "models"  # astar_island/models/
    # If DATA_DIR is relative, resolve from cwd
    if not data_dir.is_absolute():
        cwd = Path.cwd()
        data_dir = cwd / data_dir
        models_dir = cwd / "astar_island" / "models"

    client = AstarClient()
    cnn_model, lgbm_model = _load_models(models_dir)

    # ── Get round ──────────────────────────────────────────────────────
    if args.round_id:
        detail = client.get_round_detail(args.round_id)
        round_id = args.round_id
    else:
        active = client.get_active_round()
        round_id = active["id"]
        detail = client.get_round_detail(round_id)

    validate_round_detail(detail)
    log.info(
        f"Round {detail['round_number']} ({round_id}) — "
        f"{detail['map_width']}x{detail['map_height']} map, "
        f"{detail['seeds_count']} seeds"
    )

    # ── Init state + store ─────────────────────────────────────────────
    state = RoundState.from_detail(detail)
    store = RoundStore(data_dir, round_id)

    # Build meta dict from detail
    meta = {
        "id": detail["id"],
        "round_number": detail["round_number"],
        "status": detail["status"],
        "map_width": detail["map_width"],
        "map_height": detail["map_height"],
        "seeds_count": detail["seeds_count"],
    }
    store.save_meta(meta)
    store.save_detail(detail)
    import numpy as np
    from astar_island.map.encoding import encode_grid
    for i, seed_state in enumerate(detail["initial_states"]):
        raw = np.array(seed_state["grid"], dtype=np.int16)
        enc = encode_grid(raw)
        store.save_raw_grid(i, raw)
        store.save_encoded_grid(i, enc)

    log.info("State initialised and initial grids persisted")

    config_snapshot = {
        "mode": args.mode,
        "model": "cnn" if cnn_model is not None else ("lgbm" if lgbm_model is not None else "baseline"),
        "budget": 50,
    }

    # ── SUBMIT 1: Baseline (prior-only, no queries) ────────────────────
    log.info("Inferring baseline prediction (no queries)...")
    pred = infer_full_map(state, lgbm_model=lgbm_model, cnn_model=cnn_model)
    submit_all_seeds(
        client, round_id, pred, state, store,
        model_version="baseline_pre_queries",
        config_snapshot=config_snapshot,
        dry_run=dry_run,
    )
    log.info("Submit 1 done")

    # ── Phase 1: Coverage queries ──────────────────────────────────────
    policy = QueryPolicy(state, budget=50)
    coverage_specs = policy.coverage_queries()
    log.info(f"Phase 1: executing {len(coverage_specs)} coverage queries...")
    _execute_queries(client, state, store, coverage_specs, dry_run)

    # ── SUBMIT 2: After coverage ───────────────────────────────────────
    log.info("Inferring after coverage queries...")
    pred = infer_full_map(state, lgbm_model=lgbm_model, cnn_model=cnn_model)
    submit_all_seeds(
        client, round_id, pred, state, store,
        model_version="posterior_post_coverage",
        config_snapshot=config_snapshot,
        dry_run=dry_run,
    )
    log.info("Submit 2 done")

    # ── Phase 2: Adaptive queries on remaining budget ──────────────────
    queries_used = len(coverage_specs) if not dry_run else 0
    remaining = max(0, 50 - queries_used)
    if remaining > 0:
        adaptive_specs = policy.adaptive_queries(remaining)
        log.info(f"Phase 2: executing {len(adaptive_specs)} adaptive queries...")
        _execute_queries(client, state, store, adaptive_specs, dry_run)

    # ── SUBMIT 3: Final ────────────────────────────────────────────────
    log.info("Final inference...")
    pred = infer_full_map(state, lgbm_model=lgbm_model, cnn_model=cnn_model)
    submit_all_seeds(
        client, round_id, pred, state, store,
        model_version="posterior_final",
        config_snapshot=config_snapshot,
        dry_run=dry_run,
    )
    log.info("Submit 3 done — round complete")

    # ── Summary ────────────────────────────────────────────────────────
    for s in range(state.seeds_count):
        log.info(
            f"Seed {s}: coverage={state.coverage_fraction(s):.1%} "
            f"queries_used={state.queries_used}"
        )

    from astar_island.utils.io import save_json
    save_json(
        data_dir / "rounds" / round_id / "run_summary.json",
        {
            "round_id": round_id,
            "round_number": detail["round_number"],
            "mode": args.mode,
            "queries_executed": state.queries_used if not dry_run else 0,
            "model": "lgbm" if lgbm_model else "baseline",
            "seed_coverage": [state.coverage_fraction(s) for s in range(state.seeds_count)],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Astar Island online round")
    parser.add_argument(
        "--mode", choices=["live", "dryrun"], default="live",
        help="live = submit to API; dryrun = validate only",
    )
    parser.add_argument(
        "--round_id", default=None,
        help="Specific round ID (default: use active round)",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
