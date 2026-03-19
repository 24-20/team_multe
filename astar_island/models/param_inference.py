"""Hidden parameter inference from viewport observations.

The simulator is controlled by hidden parameters that are SHARED across all 5 seeds
in a round.  This means every observation — regardless of which seed it came from —
teaches us something about the parameters, and we can use that knowledge to improve
predictions on seeds we haven't even queried yet.

The 6 parameters we infer
--------------------------
winter_severity   [0, 1]
    How brutal the annual winters are.  High severity → many settlements starve and
    collapse into ruins.  Signal: fraction of observed initial settlements that died.

expansion_rate    [0, 1]
    How aggressively thriving settlements found new outposts.  High rate → many new
    settlements appear on previously empty/forest land near existing ones.
    Signal: new-settlement density in observed viewports.

raid_intensity    [0, 1]
    How often and how hard settlements raid neighbours.  High intensity → conquered
    settlements flip allegiance or collapse.  Hard to separate from winter without
    faction data, but raided collapses tend to appear near other settlements while
    winter collapses are more uniformly distributed.
    Signal: collapse rate conditioned on local settlement density.

ruin_recovery     [0, 1]
    How quickly abandoned ruins are reclaimed by nearby settlements.  High recovery →
    ruins rarely persist for long; they turn back into settlements (or ports on coast).
    Signal: fraction of observed initial ruins that are now settlement/port.

port_development  [0, 1]
    How readily coastal settlements build harbours.  High development → many coastal
    non-port settlements gain port status; ports are retained even under pressure.
    Signal: fraction of observed coastal settlements that have port status post-sim.

forest_growth     [0, 1]
    How quickly abandoned land is reclaimed by forest.  High growth → ruins and empty
    cells far from settlements fill with trees.
    Signal: fraction of observed initially-empty land that is now forest.

Uncertainty handling
--------------------
With only ~10 observations per seed and stochastic outcomes, estimates are noisy.
We weight every adjustment by a *confidence* factor that grows with sample size:

    confidence(n) = 1 - exp(-n / N_HALF)

where N_HALF = 5 (reaches 63% confidence at 5 observations, 86% at 10).
With 0 observations, no adjustment is made.  With 20+, the estimate dominates.

This prevents the model from being overconfident from a single lucky observation.
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field

import numpy as np

from ..world.terrain import TerrainCode, TERRAIN_TO_CLASS, SeedWorld, WorldState
from ..obs.planner import ObservationStore, Observation

logger = logging.getLogger(__name__)

# How many observations give ~63% confidence (one "half-life")
_N_HALF = 5.0


def _confidence(n: int) -> float:
    """Confidence weight that grows with sample size, saturates near 1."""
    return 1.0 - math.exp(-n / _N_HALF)


@dataclass
class HiddenParams:
    """Posterior estimates of the round's hidden parameters.

    Each field is in [0, 1].  0.5 = prior (no information).
    Confidence tracks how many relevant observations back the estimate.
    """
    winter_severity:  float = 0.5
    expansion_rate:   float = 0.5
    raid_intensity:   float = 0.5
    ruin_recovery:    float = 0.5
    port_development: float = 0.5
    forest_growth:    float = 0.5

    # Sample sizes driving each estimate
    n_survival:   int = 0   # initial settlements observed alive or dead
    n_expansion:  int = 0   # empty cells observed for new-settlement signal
    n_ruin:       int = 0   # initial ruins observed
    n_port:       int = 0   # coastal settlements observed
    n_forest:     int = 0   # empty/plains cells observed

    @property
    def survival_rate(self) -> float:
        """1 - winter_severity, for convenience."""
        return 1.0 - self.winter_severity

    def confidence_summary(self) -> dict:
        return {
            "winter_severity":  (self.winter_severity,  _confidence(self.n_survival)),
            "expansion_rate":   (self.expansion_rate,   _confidence(self.n_expansion)),
            "raid_intensity":   (self.raid_intensity,   _confidence(self.n_survival)),
            "ruin_recovery":    (self.ruin_recovery,    _confidence(self.n_ruin)),
            "port_development": (self.port_development, _confidence(self.n_port)),
            "forest_growth":    (self.forest_growth,    _confidence(self.n_forest)),
        }

    def log_summary(self) -> None:
        for name, (val, conf) in self.confidence_summary().items():
            logger.info("  %-20s = %.2f  (confidence %.0f%%)", name, val, conf * 100)


def infer_params(
    world: WorldState,
    store: ObservationStore,
) -> HiddenParams:
    """Estimate hidden parameters from ALL observations across ALL seeds.

    Hidden parameters are round-level (shared across seeds), so we pool every
    observation together regardless of seed index.
    """
    round_id = world.round_id
    all_obs = store.load_all(round_id)

    if not all_obs:
        logger.info("No observations yet — using uninformative prior (0.5 for all params)")
        return HiddenParams()

    # Aggregate counts across all seeds and viewports
    # We need the initial state for each seed to distinguish "initial" vs "new" cells
    initial_settlements: dict[int, set[tuple[int, int]]] = {}
    initial_ports:       dict[int, set[tuple[int, int]]] = {}
    initial_ruins:       dict[int, set[tuple[int, int]]] = {}
    for s in world.seeds:
        initial_settlements[s.seed_index] = set(s.settlement_positions)
        initial_ports[s.seed_index]       = set(s.port_positions)
        initial_ruins[s.seed_index]       = {
            (x, y)
            for y in range(s.height)
            for x in range(s.width)
            if s.terrain[y, x] == TerrainCode.RUIN
        }

    # --- Counters ---
    # Settlement survival
    initial_seen_alive = 0
    initial_seen_dead  = 0

    # Expansion: empty/plains cells that became settlement/port post-sim
    empty_became_settle = 0
    empty_stayed_empty  = 0

    # Raid intensity: collapses that occurred near other settlements
    collapse_near_settle  = 0
    collapse_far_settle   = 0

    # Ruin recovery
    ruin_recovered = 0
    ruin_stayed    = 0

    # Port development: coastal non-port settlements that got ports
    coastal_nosettled_got_port   = 0
    coastal_nosettled_no_port    = 0

    # Forest growth: initially empty land that became forest
    empty_became_forest = 0
    empty_total_checked = 0

    for obs in all_obs:
        si = obs.seed_index
        sw = world.seed(si)

        for wx, wy, post_code in obs.iter_cells():
            if wx >= sw.width or wy >= sw.height:
                continue

            pre_code = int(sw.terrain[wy, wx])
            post_class = TERRAIN_TO_CLASS.get(post_code)
            is_coastal = bool(sw.coast_mask[wy, wx])
            dist = float(sw.settlement_distance[wy, wx])
            density = int(sw.local_settlement_density[wy, wx])
            is_initial_settle = (wx, wy) in initial_settlements[si]
            is_initial_port   = (wx, wy) in initial_ports[si]
            is_initial_ruin   = (wx, wy) in initial_ruins[si]

            # --- Settlement survival ---
            if is_initial_settle or is_initial_port:
                if post_class in (1, 2):  # still settlement or port
                    initial_seen_alive += 1
                elif post_class == 3:     # became ruin
                    initial_seen_dead += 1
                    # Did it collapse near other settlements (raid) or in isolation (winter)?
                    if density >= 2:
                        collapse_near_settle += 1
                    else:
                        collapse_far_settle += 1
                # If post_class == 0 (empty/plains) it fully dissolved — count as dead
                elif post_class == 0:
                    initial_seen_dead += 1

            # --- Ruin recovery ---
            elif is_initial_ruin:
                if post_class in (1, 2):
                    ruin_recovered += 1
                else:
                    ruin_stayed += 1

            # --- Expansion / forest growth (on initially empty land) ---
            elif pre_code in (TerrainCode.PLAINS, TerrainCode.EMPTY):
                empty_total_checked += 1
                if post_class == 1 or post_class == 2:
                    empty_became_settle += 1
                elif post_class == 4:
                    empty_became_forest += 1
                else:
                    empty_stayed_empty += 1

                # Port development: coastal non-port settlements
                if is_initial_settle and not is_initial_port and is_coastal:
                    if post_class == 2:
                        coastal_nosettled_got_port += 1
                    elif post_class in (1, 3):
                        coastal_nosettled_no_port += 1

    # --- Convert counts to parameter estimates ---
    params = HiddenParams()

    # Winter severity + raid intensity (both kill settlements)
    n_survival = initial_seen_alive + initial_seen_dead
    params.n_survival = n_survival
    if n_survival > 0:
        survival_rate = initial_seen_alive / n_survival
        # winter_severity is inversely proportional to survival
        params.winter_severity = _blend(0.5, 1.0 - survival_rate, _confidence(n_survival))

        # Raid vs winter split: if most collapses are near other settlements → raids
        n_collapses = collapse_near_settle + collapse_far_settle
        if n_collapses > 0:
            raid_fraction = collapse_near_settle / n_collapses
            # raid_intensity: high if most collapses happen near neighbours
            params.raid_intensity = _blend(0.5, raid_fraction, _confidence(n_collapses))

    # Expansion rate: fraction of empty cells that became settlements
    n_expansion = empty_became_settle + empty_stayed_empty + empty_became_forest
    params.n_expansion = n_expansion
    if n_expansion > 0:
        raw_expansion = empty_became_settle / n_expansion
        # Normalise: max plausible expansion fraction is ~0.3 (30% of empty cells)
        params.expansion_rate = _blend(0.5, min(raw_expansion / 0.3, 1.0), _confidence(n_expansion))

    # Ruin recovery
    n_ruin = ruin_recovered + ruin_stayed
    params.n_ruin = n_ruin
    if n_ruin > 0:
        params.ruin_recovery = _blend(0.5, ruin_recovered / n_ruin, _confidence(n_ruin))

    # Port development
    n_port = coastal_nosettled_got_port + coastal_nosettled_no_port
    params.n_port = n_port
    if n_port > 0:
        params.port_development = _blend(0.5, coastal_nosettled_got_port / n_port, _confidence(n_port))

    # Forest growth
    params.n_forest = empty_total_checked
    if empty_total_checked > 0:
        raw_forest = empty_became_forest / empty_total_checked
        params.forest_growth = _blend(0.5, min(raw_forest / 0.4, 1.0), _confidence(empty_total_checked))

    logger.info("Inferred hidden parameters from %d observations:", len(all_obs))
    params.log_summary()
    return params


def _blend(prior: float, estimate: float, confidence: float) -> float:
    """Interpolate between prior (0 confidence) and estimate (full confidence)."""
    return prior * (1 - confidence) + estimate * confidence


def apply_params_to_prior(
    alpha: np.ndarray,
    sw: SeedWorld,
    params: HiddenParams,
) -> np.ndarray:
    """Adjust per-cell Dirichlet alpha using inferred hidden parameters.

    For each parameter, we compute an effect magnitude scaled by confidence,
    and apply it only to cells where that parameter is relevant.
    The adjustment is additive on alpha (pseudo-counts), so it respects
    existing observation evidence — cells we've seen directly are barely affected.

    Parameters
    ----------
    alpha : [H, W, 6] Dirichlet pseudo-count array (already updated with observations)
    sw    : SeedWorld with terrain masks
    params: HiddenParams from infer_params()

    Returns
    -------
    Adjusted alpha array (same shape).
    """
    H, W = sw.height, sw.width

    # How strongly to push vs the existing alpha.
    # We use a small nudge (max 1.5 pseudo-counts) so direct observations dominate.
    MAX_NUDGE = 1.5

    # --- 1. Winter severity: initial settlements → shift settlement→ruin ---
    w_conf = _confidence(params.n_survival)
    if w_conf > 0.05:
        # How much to shift: severity 0.5 = neutral, 1.0 = brutal
        severity_delta = (params.winter_severity - 0.5) * 2   # [-1, 1]
        nudge = severity_delta * MAX_NUDGE * w_conf

        settle_mask = (sw.terrain == TerrainCode.SETTLEMENT) | (sw.terrain == TerrainCode.PORT)
        # Positive nudge (harsher) → more ruin, less settlement/port
        alpha[settle_mask, 3] += max(nudge, 0)
        alpha[settle_mask, 1] -= max(nudge, 0) * 0.6
        alpha[settle_mask, 2] -= max(nudge, 0) * 0.4
        # Negative nudge (milder) → more settlement, less ruin
        alpha[settle_mask, 1] += max(-nudge, 0) * 0.6
        alpha[settle_mask, 2] += max(-nudge, 0) * 0.4
        alpha[settle_mask, 3] -= max(-nudge, 0)

    # --- 2. Expansion rate: empty cells near settlements → shift empty→settlement ---
    e_conf = _confidence(params.n_expansion)
    if e_conf > 0.05:
        expansion_delta = (params.expansion_rate - 0.5) * 2
        nudge = expansion_delta * MAX_NUDGE * e_conf

        empty_near = (
            (sw.terrain == TerrainCode.PLAINS) | (sw.terrain == TerrainCode.EMPTY)
        ) & (sw.settlement_distance <= 6)

        alpha[empty_near, 1] += max(nudge, 0)
        alpha[empty_near, 0] -= max(nudge, 0)
        alpha[empty_near, 0] += max(-nudge, 0)
        alpha[empty_near, 1] -= max(-nudge, 0)

    # --- 3. Ruin recovery: ruin cells → shift ruin→settlement ---
    r_conf = _confidence(params.n_ruin)
    if r_conf > 0.05:
        recovery_delta = (params.ruin_recovery - 0.5) * 2
        nudge = recovery_delta * MAX_NUDGE * r_conf

        ruin_mask = sw.terrain == TerrainCode.RUIN
        ruin_near = ruin_mask & (sw.settlement_distance <= 8)

        alpha[ruin_near, 1] += max(nudge, 0) * 0.6
        alpha[ruin_near, 2] += max(nudge, 0) * 0.4 * (sw.coast_mask[ruin_near].mean() if ruin_near.any() else 0)
        alpha[ruin_near, 3] -= max(nudge, 0)
        alpha[ruin_near, 3] += max(-nudge, 0)
        alpha[ruin_near, 1] -= max(-nudge, 0)

    # --- 4. Port development: coastal settlements → shift settlement→port ---
    p_conf = _confidence(params.n_port)
    if p_conf > 0.05:
        port_delta = (params.port_development - 0.5) * 2
        nudge = port_delta * MAX_NUDGE * p_conf

        coastal_settle = (sw.terrain == TerrainCode.SETTLEMENT) & sw.coast_mask

        alpha[coastal_settle, 2] += max(nudge, 0)
        alpha[coastal_settle, 1] -= max(nudge, 0)
        alpha[coastal_settle, 1] += max(-nudge, 0)
        alpha[coastal_settle, 2] -= max(-nudge, 0)

    # --- 5. Forest growth: empty land far from settlements → shift empty→forest ---
    f_conf = _confidence(params.n_forest)
    if f_conf > 0.05:
        forest_delta = (params.forest_growth - 0.5) * 2
        nudge = forest_delta * MAX_NUDGE * f_conf

        empty_far = (
            (sw.terrain == TerrainCode.PLAINS) | (sw.terrain == TerrainCode.EMPTY)
        ) & (sw.settlement_distance > 8)

        alpha[empty_far, 4] += max(nudge, 0)
        alpha[empty_far, 0] -= max(nudge, 0)
        alpha[empty_far, 0] += max(-nudge, 0)
        alpha[empty_far, 4] -= max(-nudge, 0)

    # Clamp alpha to avoid negatives (shouldn't happen, but guard anyway)
    np.clip(alpha, 1e-6, None, out=alpha)
    return alpha
