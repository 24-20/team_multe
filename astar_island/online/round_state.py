"""RoundState — live state during an active round.

Design principles:
- raw_grids and encoded_grids are stored separately (raw needed for static masks)
- empirical_counts (from queries) and prior_probs (from initial grid) are kept separate
- posterior_mean is computed on demand as a Bayesian blend of the two
- All coordinate inputs in update_from_query use API convention (x, y);
  conversion to internal (row, col) happens immediately inside this class
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..map.encoding import encode_grid, build_prior_map
from ..validators import validate_round_detail


_DEFAULT_PRIOR_STRENGTH = 2.0


@dataclass
class RoundState:
    round_id: str
    width: int
    height: int
    seeds_count: int

    # [seeds, H, W] — raw API terrain codes (needed for static mask checks)
    raw_grids: np.ndarray
    # [seeds, H, W] — encoded prediction classes 0-5
    encoded_grids: np.ndarray

    # [seeds, H, W, 6] — per-cell class prior from initial terrain (float64)
    prior_probs: np.ndarray
    # [seeds, H, W, 6] — accumulated observation counts from queries (float64)
    empirical_counts: np.ndarray
    # [seeds, H, W] — how many times each cell has been observed (int32)
    observed_count: np.ndarray

    # Per-seed settlement info from initial states
    initial_settlements: list[list[dict]]

    # Log of all executed queries
    query_log: list[dict] = field(default_factory=list)
    queries_used: int = 0

    # Prior strength: alpha = prior_probs * prior_strength
    prior_strength: float = _DEFAULT_PRIOR_STRENGTH

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_detail(
        cls,
        detail: dict,
        prior_strength: float = _DEFAULT_PRIOR_STRENGTH,
    ) -> "RoundState":
        """Build a fresh RoundState from a round detail API response."""
        validate_round_detail(detail)

        round_id = detail["id"]
        W = detail["map_width"]
        H = detail["map_height"]
        seeds = detail["seeds_count"]

        raw_grids = np.zeros((seeds, H, W), dtype=np.int16)
        encoded_grids = np.zeros((seeds, H, W), dtype=np.int8)
        prior_probs = np.zeros((seeds, H, W, 6), dtype=np.float64)
        initial_settlements: list[list[dict]] = []

        for i, state in enumerate(detail["initial_states"]):
            raw = np.array(state["grid"], dtype=np.int16)  # [H, W]
            raw_grids[i] = raw
            encoded_grids[i] = encode_grid(raw)
            prior_probs[i] = build_prior_map(raw)
            initial_settlements.append(state.get("settlements", []))

        empirical_counts = np.zeros((seeds, H, W, 6), dtype=np.float64)
        observed_count = np.zeros((seeds, H, W), dtype=np.int32)

        return cls(
            round_id=round_id,
            width=W,
            height=H,
            seeds_count=seeds,
            raw_grids=raw_grids,
            encoded_grids=encoded_grids,
            prior_probs=prior_probs,
            empirical_counts=empirical_counts,
            observed_count=observed_count,
            initial_settlements=initial_settlements,
            prior_strength=prior_strength,
        )

    # ------------------------------------------------------------------
    # Query update
    # ------------------------------------------------------------------

    def update_from_query(
        self,
        seed_idx: int,
        viewport_x: int,
        viewport_y: int,
        viewport_w: int,
        viewport_h: int,
        grid_result: list[list[int]],
        settlements_result: list[dict] | None = None,
        query_metadata: dict | None = None,
    ) -> None:
        """Integrate one simulate() response into the round state.

        API uses (x=col, y=row) convention. We convert to (row, col) immediately.

        Args:
            seed_idx:         seed index (0-4)
            viewport_x:       left column of viewport (API x)
            viewport_y:       top row of viewport (API y)
            viewport_w/h:     viewport dimensions
            grid_result:      viewport_h x viewport_w raw terrain codes
            settlements_result: optional list of settlement dicts in viewport
            query_metadata:   arbitrary dict stored in query_log
        """
        # API: x = col, y = row
        row_start = viewport_y   # y → row
        col_start = viewport_x   # x → col

        raw_vp = np.array(grid_result, dtype=np.int16)  # [vp_h, vp_w]
        from ..map.encoding import encode_grid, RAW_TO_CLASS
        enc_vp = encode_grid(raw_vp)  # [vp_h, vp_w]

        for dr in range(viewport_h):
            for dc in range(viewport_w):
                row = row_start + dr
                col = col_start + dc
                if 0 <= row < self.height and 0 <= col < self.width:
                    cls = int(enc_vp[dr, dc])
                    self.empirical_counts[seed_idx, row, col, cls] += 1.0
                    self.observed_count[seed_idx, row, col] += 1

        # Store raw terrain in raw_grids if observed (helps with per-query replay)
        # Note: stochastic run — don't overwrite static cells with query noise
        from ..map.encoding import STATIC_RAW_CODES
        for dr in range(viewport_h):
            for dc in range(viewport_w):
                row = row_start + dr
                col = col_start + dc
                if 0 <= row < self.height and 0 <= col < self.width:
                    raw_code = int(raw_vp[dr, dc])
                    if raw_code in STATIC_RAW_CODES:
                        self.raw_grids[seed_idx, row, col] = raw_code

        log_entry: dict[str, Any] = {
            "seed_idx": seed_idx,
            "viewport_x": viewport_x,
            "viewport_y": viewport_y,
            "viewport_w": viewport_w,
            "viewport_h": viewport_h,
            "metadata": query_metadata or {},
        }
        if settlements_result:
            log_entry["settlements"] = settlements_result
        self.query_log.append(log_entry)
        self.queries_used += 1

    # ------------------------------------------------------------------
    # Posterior
    # ------------------------------------------------------------------

    @property
    def posterior_mean(self) -> np.ndarray:
        """Compute Bayesian posterior mean: [seeds, H, W, 6].

        For observed cells:
            posterior = (empirical_counts + alpha) / (observed_count + sum(alpha))
            where alpha = prior_probs * prior_strength

        For unobserved cells:
            If dynamism_rate is high, blend the static cell prior toward the
            round's global observed distribution. This corrects for rounds
            where the simulation has moved many cells away from their initial
            state — in those rounds the initial prior is misleading.

            blend = clip(dynamism_rate * 3, 0, 0.6)
            adjusted = (1 - blend) * cell_prior + blend * global_dist
        """
        alpha = self.prior_probs * self.prior_strength        # [s, H, W, 6]
        alpha_sum = alpha.sum(axis=-1, keepdims=True)          # [s, H, W, 1]
        n = self.observed_count[..., np.newaxis].astype(np.float64)  # [s, H, W, 1]

        num = self.empirical_counts + alpha
        denom = n + alpha_sum
        posterior = num / denom
        posterior /= posterior.sum(axis=-1, keepdims=True)

        # Adjust unobserved cells using round-global distribution
        for s in range(self.seeds_count):
            rf = self.round_features(s)
            dyn = rf["dynamism_rate"]
            if dyn < 0.02:
                continue  # not enough signal — keep static prior
            blend = min(dyn * 3.0, 0.6)
            global_dist = self.global_obs_distribution(s)  # [6]
            unobs = self.observed_count[s] == 0  # [H, W]
            if not unobs.any():
                continue
            cell_prior = self.prior_probs[s, unobs]  # [n_unobs, 6]
            adjusted = (1.0 - blend) * cell_prior + blend * global_dist
            adjusted /= adjusted.sum(axis=-1, keepdims=True)
            posterior[s, unobs] = adjusted

        return posterior

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def coverage_fraction(self, seed_idx: int) -> float:
        """Fraction of map cells observed at least once for this seed."""
        observed = (self.observed_count[seed_idx] > 0).sum()
        return float(observed) / (self.height * self.width)

    def round_features(self, seed_idx: int) -> dict:
        """Aggregate round-level statistics from query observations."""
        ec = self.empirical_counts[seed_idx]  # [H, W, 6]
        total = ec.sum()
        if total < 1:
            return {
                "ruin_rate": 0.0,
                "port_rate": 0.0,
                "settlement_survival": 1.0,
                "dynamism_rate": 0.0,
                "settlement_density": 0.0,
                "empty_rate": 1.0,
            }
        ruin_rate = float(ec[..., 3].sum() / total)
        port_rate = float(ec[..., 2].sum() / total)
        sett_total = float(ec[..., 1].sum() + ec[..., 3].sum())
        survival = float(ec[..., 1].sum() / max(sett_total, 1))

        # Dynamism: fraction of observed cells whose most-likely class differs
        # from the initial encoded_grid class (cell has changed from initial state)
        obs_mask = self.observed_count[seed_idx] > 0  # [H, W]
        n_obs = obs_mask.sum()
        if n_obs > 0:
            inferred_class = ec[obs_mask].argmax(axis=-1)  # [n_obs]
            initial_class = self.encoded_grids[seed_idx][obs_mask]  # [n_obs]
            dynamism_rate = float((inferred_class != initial_class).sum() / n_obs)
        else:
            dynamism_rate = 0.0

        settlement_density = float(ec[..., 1].sum() / total)
        empty_rate = float(ec[..., 0].sum() / total)

        return {
            "ruin_rate": ruin_rate,
            "port_rate": port_rate,
            "settlement_survival": survival,
            "dynamism_rate": dynamism_rate,
            "settlement_density": settlement_density,
            "empty_rate": empty_rate,
        }

    def global_obs_distribution(self, seed_idx: int) -> np.ndarray:
        """Return [6] normalized class distribution across all observed cells.

        Falls back to uniform if nothing observed yet.
        """
        ec = self.empirical_counts[seed_idx]  # [H, W, 6]
        global_counts = ec.sum(axis=(0, 1))   # [6]
        total = global_counts.sum()
        if total < 1:
            return np.full(6, 1.0 / 6, dtype=np.float64)
        return global_counts / total

    def dynamism_map(self, seed_idx: int) -> np.ndarray:
        """Return [H, W] float array: 1.0 where observed class != initial class, else 0.

        Only observed cells can be non-zero.
        """
        ec = self.empirical_counts[seed_idx]  # [H, W, 6]
        obs_mask = self.observed_count[seed_idx] > 0
        inferred = ec.argmax(axis=-1)  # [H, W]
        initial = self.encoded_grids[seed_idx]  # [H, W]
        changed = (inferred != initial).astype(np.float64)
        changed[~obs_mask] = 0.0  # unobserved cells are not known to have changed
        return changed
