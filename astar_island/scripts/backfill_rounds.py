"""Backfill all historical rounds from the API.

Usage:
    python -m astar_island.scripts.backfill_rounds [--build-dataset]
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
from astar_island.ingest.post_round import _fetch_analysis
from astar_island.utils.io import ensure_dir, save_json
from astar_island.utils.logging import get_logger
from astar_island.map.encoding import encode_grid

import numpy as np

log = get_logger("backfill_rounds")


def backfill_round(
    client: AstarClient,
    round_meta: dict,
    data_dir: Path,
) -> None:
    round_id = round_meta["id"]
    store = RoundStore(data_dir, round_id)

    if store.meta_exists():
        log.info(f"Round {round_id} already backfilled — skipping meta fetch")
    else:
        log.info(f"Backfilling round {round_id}...")
        try:
            detail = client.get_round_detail(round_id)
        except Exception as e:
            log.error(f"Failed to get detail for {round_id}: {e}")
            return

        store.save_meta(round_meta)
        store.save_detail(detail)

        for i, seed_state in enumerate(detail.get("initial_states", [])):
            raw = np.array(seed_state["grid"], dtype=np.int16)
            enc = encode_grid(raw)
            store.save_raw_grid(i, raw)
            store.save_encoded_grid(i, enc)

    # Fetch analysis for completed rounds
    try:
        detail = store.load_detail()
        status = detail.get("status", "")
        seeds_count = detail.get("seeds_count", 5)
    except Exception:
        return

    if status == "completed":
        log.info(f"Fetching analysis for completed round {round_id}...")
        _fetch_analysis(client, round_id, store, seeds_count)

    # Fetch predictions if available
    try:
        predictions = client.my_predictions(round_id)
        save_json(store.root / "my_predictions.json", predictions)
    except Exception as e:
        log.warning(f"Could not fetch predictions for {round_id}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill all historical rounds")
    parser.add_argument("--build-dataset", action="store_true",
                        help="Build training dataset after backfill")
    args = parser.parse_args()

    data_dir = Path(os.environ.get("DATA_DIR", "astar_island/data"))
    if not data_dir.is_absolute():
        data_dir = Path.cwd() / data_dir
    ensure_dir(data_dir / "rounds")

    client = AstarClient()

    log.info("Fetching all rounds from my-rounds...")
    try:
        my_rounds = client.my_rounds()
    except Exception as e:
        log.error(f"Failed to fetch my_rounds: {e}")
        return

    log.info(f"Found {len(my_rounds)} rounds to backfill")
    for r in my_rounds:
        backfill_round(client, r, data_dir)

    if args.build_dataset:
        log.info("Building training dataset...")
        from astar_island.training.build_dataset import build_cell_dataset
        df = build_cell_dataset(data_dir)
        if not df.empty:
            log.info(f"Dataset built: {len(df)} rows")
        else:
            log.info("Dataset is empty (no completed rounds with analysis yet)")


if __name__ == "__main__":
    main()
