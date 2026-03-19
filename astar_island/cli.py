"""CLI for the Astar Island competition pipeline.

Commands:
    observe   — Run the observation loop (query the simulator)
    predict   — Generate predictions from stored observations (no API calls)
    submit    — Generate predictions and submit to the API
    run       — Full pipeline: observe → predict → submit
    score     — Local scoring against a provided ground truth file

Usage:
    python -m astar_island.cli run --token YOUR_JWT [--round-id ROUND_ID] [--dry-run]
    python -m astar_island.cli observe --token YOUR_JWT [--round-id ROUND_ID] [--budget 20]
    python -m astar_island.cli predict [--round-id ROUND_ID] [--data-dir data]
    python -m astar_island.cli submit --token YOUR_JWT [--round-id ROUND_ID] [--dry-run]

Authentication:
    Log in at app.ainm.no, inspect browser cookies, copy the access_token JWT.
    Pass it via --token or set ASTAR_TOKEN in your .env file.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Allow running as `python -m astar_island.cli` from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from astar_island.api.client import AstarClient, Round, QueryBudgetExhausted
from astar_island.world.terrain import WorldState
from astar_island.obs.planner import ObservationStore, ViewportPlanner, Observation
from astar_island.models.predictor import RuleBasedPredictor, prediction_to_list, apply_probability_floor
from astar_island.eval.scorer import local_score_report, validate_prediction_tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_token(args: argparse.Namespace) -> str:
    token = getattr(args, "token", None) or os.environ.get("ASTAR_TOKEN", "")
    if not token:
        logger.error("No token provided. Use --token or set ASTAR_TOKEN in .env")
        sys.exit(1)
    return token


def _resolve_round_id(client: AstarClient, args: argparse.Namespace) -> str:
    round_id = getattr(args, "round_id", None)
    if round_id:
        return round_id
    active = client.get_active_round_id()
    if active is None:
        logger.error("No active round found. Specify --round-id explicitly.")
        sys.exit(1)
    logger.info("Auto-detected active round: %s", active)
    return active


def _load_or_fetch_round(
    client: AstarClient | None,
    store: ObservationStore,
    round_id: str,
) -> Round:
    """Load round detail from cache or fetch from API."""
    cached = store.load_round_detail(round_id)
    if cached:
        logger.info("Using cached round detail for %s", round_id)
        return Round.from_dict(cached)
    if client is None:
        logger.error("Round detail not cached and no client available.")
        sys.exit(1)
    detail = client._request("GET", f"/astar-island/rounds/{round_id}")
    store.save_round_detail(round_id, detail)
    return Round.from_dict(detail)


def _save_prediction(round_id: str, seed_index: int, probs: np.ndarray, data_dir: Path) -> Path:
    pred_dir = data_dir / round_id / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    path = pred_dir / f"seed_{seed_index}.json"
    path.write_text(json.dumps(prediction_to_list(probs)))
    logger.info("Saved prediction for seed %d → %s", seed_index, path)
    return path


def _load_prediction(round_id: str, seed_index: int, data_dir: Path) -> np.ndarray | None:
    path = data_dir / round_id / "predictions" / f"seed_{seed_index}.json"
    if not path.exists():
        return None
    return np.array(json.loads(path.read_text()))


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def cmd_observe(args: argparse.Namespace) -> None:
    """Run observation queries against the simulator."""
    token = _get_token(args)
    data_dir = Path(args.data_dir)
    budget = args.budget

    client = AstarClient(token=token, budget=budget)
    store = ObservationStore(data_dir)

    round_id = _resolve_round_id(client, args)
    round_obj = _load_or_fetch_round(client, store, round_id)
    world = WorldState.from_round(round_obj)

    already_used = store.query_count(round_id)
    effective_budget = budget - already_used
    if effective_budget <= 0:
        logger.info("All %d queries already used for round %s", budget, round_id)
        return

    logger.info(
        "Round %d: %dx%d, %d seeds, %d queries remaining (budget=%d, used=%d)",
        round_obj.round_number, round_obj.map_width, round_obj.map_height,
        round_obj.seeds_count, effective_budget, budget, already_used,
    )

    planner = ViewportPlanner(world, store, total_budget=effective_budget)
    plan = planner.plan_round()

    for seed_idx, vx, vy, vw, vh in plan:
        if client.budget_remaining <= 0:
            logger.info("Budget exhausted after %d queries", client.budget_used)
            break
        try:
            result = client.simulate(round_id, seed_idx, vx, vy, vw, vh)
        except QueryBudgetExhausted:
            logger.info("Budget exhausted.")
            break

        obs = Observation(
            round_id=round_id,
            seed_index=seed_idx,
            viewport_x=result.viewport.x,
            viewport_y=result.viewport.y,
            viewport_w=result.viewport.w,
            viewport_h=result.viewport.h,
            grid=result.grid,
            settlements=[
                {
                    "x": s.x, "y": s.y,
                    "has_port": s.has_port, "alive": s.alive,
                    "population": s.population, "food": s.food,
                    "wealth": s.wealth, "defense": s.defense,
                }
                for s in result.settlements
            ],
        )
        store.save(obs)
        logger.info(
            "Observed seed=%d viewport=(%d,%d)+%dx%d — %d settlements visible",
            seed_idx, vx, vy, vw, vh, len(result.settlements),
        )

    stats = store.round_level_stats(round_id)
    logger.info(
        "Round-level stats: survival_rate=%.2f  total_seen=%d",
        stats["survival_rate"], stats["total_initial_seen"],
    )


def cmd_predict(args: argparse.Namespace) -> None:
    """Generate predictions from stored observations (no API calls)."""
    data_dir = Path(args.data_dir)
    store = ObservationStore(data_dir)

    round_id = args.round_id
    if not round_id:
        logger.error("--round-id required for predict command")
        sys.exit(1)

    round_obj = _load_or_fetch_round(None, store, round_id)
    world = WorldState.from_round(round_obj)
    from astar_island.models.param_inference import infer_params
    predictor = RuleBasedPredictor(world, store)

    # Infer hidden params once from all cross-seed observations, then share
    params = infer_params(world, store)
    logger.info("Hidden parameter estimates ready")

    use_surrogate = getattr(args, "surrogate", False)
    n_runs = getattr(args, "surrogate_runs", 200)

    for seed_idx in range(round_obj.seeds_count):
        if use_surrogate:
            logger.info("Using surrogate simulator for seed %d (%d runs)", seed_idx, n_runs)
            probs = predictor.predict_with_surrogate(seed_idx, params=params, n_runs=n_runs)
        else:
            probs = predictor.predict(seed_idx, params=params)
        errors = validate_prediction_tensor(probs)
        if errors:
            logger.error("Prediction validation failed for seed %d: %s", seed_idx, errors)
            sys.exit(1)
        _save_prediction(round_id, seed_idx, probs, data_dir)
        logger.info(
            "Seed %d prediction ready — shape=%s, min_prob=%.4f",
            seed_idx, probs.shape, probs.min(),
        )


def cmd_submit(args: argparse.Namespace) -> None:
    """Submit predictions to the API."""
    token = _get_token(args)
    data_dir = Path(args.data_dir)
    dry_run = args.dry_run

    client = AstarClient(token=token)
    store = ObservationStore(data_dir)

    round_id = _resolve_round_id(client, args)
    round_obj = _load_or_fetch_round(client, store, round_id)
    world = WorldState.from_round(round_obj)
    predictor = RuleBasedPredictor(world, store)

    for seed_idx in range(round_obj.seeds_count):
        # Use cached prediction if available, otherwise generate fresh
        probs = _load_prediction(round_id, seed_idx, data_dir)
        if probs is None:
            logger.info("No cached prediction for seed %d — generating...", seed_idx)
            probs = predictor.predict(seed_idx)
            _save_prediction(round_id, seed_idx, probs, data_dir)

        errors = validate_prediction_tensor(probs)
        if errors:
            logger.error("Prediction invalid for seed %d: %s", seed_idx, errors)
            sys.exit(1)

        if dry_run:
            logger.info("[DRY RUN] Would submit seed %d shape=%s", seed_idx, probs.shape)
        else:
            resp = client.submit(round_id, seed_idx, prediction_to_list(probs))
            logger.info("Submitted seed %d — response: %s", seed_idx, resp)

    if dry_run:
        logger.info("[DRY RUN] Submission complete (nothing sent to API)")
    else:
        logger.info("All %d seeds submitted.", round_obj.seeds_count)


def cmd_run(args: argparse.Namespace) -> None:
    """Full pipeline: observe → predict → submit."""
    logger.info("=== Phase 1: Observe ===")
    cmd_observe(args)

    logger.info("=== Phase 2: Predict ===")
    # predict needs round_id; might have been auto-detected in observe — re-detect
    if not args.round_id:
        token = _get_token(args)
        client = AstarClient(token=token)
        args.round_id = client.get_active_round_id()
    cmd_predict(args)

    logger.info("=== Phase 3: Submit ===")
    cmd_submit(args)


def cmd_score(args: argparse.Namespace) -> None:
    """Compute local score against a ground truth file.

    Ground truth file format: JSON with {"seeds": [[H, W, 6 list], ...]}
    """
    data_dir = Path(args.data_dir)
    gt_path = Path(args.ground_truth)
    round_id = args.round_id

    if not gt_path.exists():
        logger.error("Ground truth file not found: %s", gt_path)
        sys.exit(1)
    if not round_id:
        logger.error("--round-id required for score command")
        sys.exit(1)

    gt_data = json.loads(gt_path.read_text())
    ground_truths = [np.array(g) for g in gt_data["seeds"]]

    predictions = []
    for i in range(len(ground_truths)):
        pred = _load_prediction(round_id, i, data_dir)
        if pred is None:
            logger.warning("No prediction for seed %d — using uniform baseline", i)
            H, W = ground_truths[i].shape[:2]
            pred = apply_probability_floor(
                np.full((H, W, 6), 1.0 / 6.0), floor=0.01
            )
        predictions.append(pred)

    report = local_score_report(ground_truths, predictions)
    print(json.dumps(report, indent=2))


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="astar_island",
        description="Astar Island competition CLI",
    )
    parser.add_argument(
        "--data-dir", default="data", metavar="DIR",
        help="Directory for storing observations and predictions (default: data)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- observe ---
    obs_p = sub.add_parser("observe", help="Query the simulator and store observations")
    obs_p.add_argument("--token", default=None, help="JWT access token (or set ASTAR_TOKEN)")
    obs_p.add_argument("--round-id", default=None)
    obs_p.add_argument("--budget", type=int, default=50, help="Total query budget (default 50)")

    # --- predict ---
    pred_p = sub.add_parser("predict", help="Generate predictions from stored observations")
    pred_p.add_argument("--round-id", default=None, required=True)
    pred_p.add_argument("--surrogate", action="store_true",
                         help="Use surrogate simulator (slower but more accurate)")
    pred_p.add_argument("--surrogate-runs", type=int, default=200, metavar="N",
                         help="Number of simulation runs for surrogate (default 200)")

    # --- submit ---
    sub_p = sub.add_parser("submit", help="Submit predictions to the API")
    sub_p.add_argument("--token", default=None)
    sub_p.add_argument("--round-id", default=None)
    sub_p.add_argument("--dry-run", action="store_true", help="Generate but do not submit")

    # --- run (full pipeline) ---
    run_p = sub.add_parser("run", help="Full pipeline: observe → predict → submit")
    run_p.add_argument("--token", default=None)
    run_p.add_argument("--round-id", default=None)
    run_p.add_argument("--budget", type=int, default=50)
    run_p.add_argument("--dry-run", action="store_true")
    run_p.add_argument("--surrogate", action="store_true",
                        help="Use surrogate simulator for predictions")
    run_p.add_argument("--surrogate-runs", type=int, default=200, metavar="N")

    # --- score ---
    score_p = sub.add_parser("score", help="Local scoring against a ground truth file")
    score_p.add_argument("--round-id", required=True)
    score_p.add_argument("--ground-truth", required=True, metavar="FILE",
                          help='JSON file: {"seeds": [[[H,W,6 tensor]], ...]}')

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    dispatch = {
        "observe": cmd_observe,
        "predict": cmd_predict,
        "submit": cmd_submit,
        "run": cmd_run,
        "score": cmd_score,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
