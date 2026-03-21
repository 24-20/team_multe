"""Fetch post-round analysis and optionally retrain model.

Usage:
    python -m astar_island.scripts.post_round_analysis --round_id ID [--retrain]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_pkg_root = Path(__file__).resolve().parents[2]
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

load_dotenv()

from astar_island.api.client import AstarClient
from astar_island.storage.store import RoundStore
from astar_island.ingest.post_round import fetch_post_round
from astar_island.utils.logging import get_logger

log = get_logger("post_round_analysis")


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-round analysis and optional retrain")
    parser.add_argument("--round_id", required=True, help="Round ID to analyse")
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain LightGBM after fetching analysis")
    parser.add_argument("--poll-interval", type=int, default=60,
                        help="Seconds between status polls (default: 60)")
    args = parser.parse_args()

    data_dir = Path(os.environ.get("DATA_DIR", "astar_island/data"))
    if not data_dir.is_absolute():
        data_dir = Path.cwd() / data_dir

    client = AstarClient()
    store = RoundStore(data_dir, args.round_id)

    try:
        detail = store.load_detail()
        seeds_count = detail.get("seeds_count", 5)
    except Exception:
        log.info("Detail not cached — fetching from API")
        detail = client.get_round_detail(args.round_id)
        seeds_count = detail.get("seeds_count", 5)

    success = fetch_post_round(
        client=client,
        round_id=args.round_id,
        store=store,
        seeds_count=seeds_count,
        poll_interval=args.poll_interval,
    )

    if not success:
        log.error("Failed to fetch analysis — aborting")
        return

    if args.retrain:
        log.info("Rebuilding dataset and retraining LightGBM...")
        from astar_island.training.build_dataset import build_cell_dataset
        from astar_island.training.train_lgbm import train_lgbm

        df = build_cell_dataset(data_dir)
        if df.empty:
            log.warning("No training data available yet")
            return

        models_dir = data_dir.parent / "models"
        if not models_dir.is_absolute():
            models_dir = Path.cwd() / "astar_island" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        model, meta = train_lgbm(df, models_dir)
        log.info(f"Retrained model — val KL={meta['val_score_kl']:.4f}")


if __name__ == "__main__":
    main()
