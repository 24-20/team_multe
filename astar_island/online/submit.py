"""Submission pipeline: validate + submit all 5 seeds."""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from ..validators import validate_prediction_tensor
from ..utils.logging import get_logger

if TYPE_CHECKING:
    from ..api.client import AstarClient
    from ..storage.store import RoundStore
    from .round_state import RoundState

log = get_logger(__name__)


def submit_all_seeds(
    client: "AstarClient",
    round_id: str,
    pred: np.ndarray,
    state: "RoundState",
    store: "RoundStore",
    model_version: str = "baseline",
    config_snapshot: dict | None = None,
    dry_run: bool = False,
) -> list[dict]:
    """Validate and submit prediction tensor for all seeds.

    Args:
        client:         AstarClient instance
        round_id:       active round ID
        pred:           [seeds, H, W, 6] prediction tensor
        state:          RoundState (for height/width)
        store:          RoundStore for persistence
        model_version:  label for metadata (e.g. "baseline", "lgbm_v1")
        config_snapshot: dict of config values to store with submission
        dry_run:        if True, skip API call and just validate + log

    Returns:
        list of API response dicts (one per seed)
    """
    responses = []
    for seed_idx in range(state.seeds_count):
        tensor = pred[seed_idx].astype(np.float64)
        try:
            validate_prediction_tensor(tensor, state.height, state.width)
        except ValueError as e:
            log.error(f"Seed {seed_idx} tensor validation failed: {e}")
            raise

        if dry_run:
            log.info(
                f"[DRYRUN] seed={seed_idx} shape={tensor.shape} "
                f"min={tensor.min():.4f} max={tensor.max():.4f} "
                f"sum_range=[{tensor.sum(-1).min():.4f}, {tensor.sum(-1).max():.4f}]"
            )
            responses.append({"dry_run": True, "seed_idx": seed_idx})
            continue

        try:
            resp = client.submit(round_id, seed_idx, tensor)
            log.info(f"Submitted seed={seed_idx}: {resp}")
        except Exception as e:
            log.error(f"Submit failed for seed {seed_idx}: {e}")
            resp = {"error": str(e), "seed_idx": seed_idx}
        time.sleep(1.0)  # extra buffer beyond token bucket — API has burst limits

        store.save_submission(
            seed_idx=seed_idx,
            response=resp,
            tensor=tensor,
            model_version=model_version,
            config_snapshot=config_snapshot or {},
        )
        responses.append(resp)

    return responses
