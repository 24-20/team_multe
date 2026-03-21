"""Post-round analysis: wait for completion and fetch ground truth."""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

from ..utils.logging import get_logger

if TYPE_CHECKING:
    from ..api.client import AstarClient
    from ..storage.store import RoundStore

log = get_logger(__name__)


def fetch_post_round(
    client: "AstarClient",
    round_id: str,
    store: "RoundStore",
    seeds_count: int = 5,
    poll_interval: int = 60,
    max_polls: int = 180,
) -> bool:
    """Poll until round is completed, then fetch and store analysis for all seeds.

    Args:
        client:        AstarClient
        round_id:      round to watch
        store:         RoundStore for this round
        seeds_count:   number of seeds to fetch
        poll_interval: seconds between polls
        max_polls:     give up after this many polls (~3 hours)

    Returns:
        True if analysis was successfully fetched, False on timeout.
    """
    log.info(f"Waiting for round {round_id} to complete...")
    for attempt in range(max_polls):
        try:
            detail = client.get_round_detail(round_id)
            status = detail.get("status", "")
        except Exception as e:
            log.warning(f"Poll {attempt}: failed to get detail: {e}")
            time.sleep(poll_interval)
            continue

        if status == "completed":
            log.info(f"Round {round_id} completed — fetching analysis")
            _fetch_analysis(client, round_id, store, seeds_count)
            return True

        log.info(f"Poll {attempt+1}/{max_polls}: status={status}, sleeping {poll_interval}s")
        time.sleep(poll_interval)

    log.warning(f"Timed out waiting for round {round_id}")
    return False


def _fetch_analysis(
    client: "AstarClient",
    round_id: str,
    store: "RoundStore",
    seeds_count: int,
) -> None:
    for seed_idx in range(seeds_count):
        if store.analysis_exists(seed_idx):
            log.info(f"Analysis seed {seed_idx} already stored — skipping")
            continue
        try:
            result = client.analysis(round_id, seed_idx)
            store.save_analysis(seed_idx, result)
            score = result.get("score", "?")
            log.info(f"Analysis seed {seed_idx}: score={score}")
        except Exception as e:
            log.error(f"Failed to fetch analysis for seed {seed_idx}: {e}")
