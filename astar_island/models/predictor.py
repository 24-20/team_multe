"""Rule-based probabilistic predictor with Bayesian observation updates.

Design (blueprint §3 + improvements):
- Stage 1: Deterministic masks — static cells get near-certain priors.
- Stage 2: Mechanism-aware heuristics — each cell's prior depends on:
    * Initial terrain type
    * Coastal status (port formation possible)
    * Distance to nearest initial settlement (expansion range)
    * Local settlement density (conflict / trade zone)
- Stage 3: Observation fusion — post-simulation terrain observations update
    the prior via a Dirichlet-multinomial posterior (soft count update).
- Stage 4: Cross-seed calibration — round-level survival stats shift priors
    uniformly so that round-specific dynamics (harsh winter, strong expansion)
    are captured even in unobserved cells.
- Stage 5: Probability floor + renormalisation (mandatory, see scoring docs).

Probability floor: always 0.01 per class, then renormalise.
Never assign 0.0 — KL divergence becomes infinite.
"""
from __future__ import annotations

import logging

import numpy as np

from ..world.terrain import TerrainCode, TERRAIN_TO_CLASS, SeedWorld, WorldState
from ..obs.planner import ObservationStore, Observation
from .param_inference import infer_params, apply_params_to_prior, HiddenParams

logger = logging.getLogger(__name__)

N_CLASSES = 6
PROB_FLOOR = 0.01

# Prior distributions: [p_empty, p_settlement, p_port, p_ruin, p_forest, p_mountain]
# Tuned to reflect typical 50-year Norse simulation dynamics.
_BASE_PRIORS: dict[str, np.ndarray] = {
    # Static cells — extremely high confidence
    "ocean":               np.array([0.960, 0.007, 0.007, 0.007, 0.012, 0.007]),
    "mountain":            np.array([0.007, 0.005, 0.005, 0.005, 0.005, 0.973]),

    # Forests mostly persist; nearby thriving settlements may absorb them
    "forest_isolated":     np.array([0.040, 0.055, 0.010, 0.080, 0.790, 0.025]),
    "forest_near_settle":  np.array([0.050, 0.120, 0.020, 0.130, 0.650, 0.030]),

    # Plains / empty — fate depends heavily on proximity to settlements
    "plains_far":          np.array([0.700, 0.060, 0.015, 0.100, 0.110, 0.015]),
    "plains_near":         np.array([0.420, 0.230, 0.050, 0.190, 0.095, 0.015]),
    "plains_coastal_far":  np.array([0.650, 0.070, 0.060, 0.100, 0.100, 0.020]),
    "plains_coastal_near": np.array([0.350, 0.200, 0.130, 0.190, 0.100, 0.030]),

    # Initial settlements: survive, collapse, or expand to ports
    "settlement_inland":   np.array([0.045, 0.490, 0.045, 0.370, 0.035, 0.015]),
    "settlement_coastal":  np.array([0.040, 0.280, 0.340, 0.290, 0.030, 0.020]),

    # Initial ports: strong port prior, possible collapse or loss of port status
    "port":                np.array([0.040, 0.150, 0.530, 0.240, 0.025, 0.015]),

    # Ruins: may be reclaimed by nearby settlements or decay to plains/forest
    "ruin_isolated":       np.array([0.290, 0.050, 0.015, 0.430, 0.200, 0.015]),
    "ruin_near_settle":    np.array([0.130, 0.180, 0.060, 0.500, 0.110, 0.020]),
    "ruin_coastal":        np.array([0.100, 0.140, 0.120, 0.510, 0.100, 0.030]),
}

# Effective prior strength (pseudo-observation count).  Higher = more resistant
# to being overridden by a small number of direct observations.
_PRIOR_STRENGTH = 4.0


def _prior_key(
    terrain_code: int,
    is_coastal: bool,
    dist_to_settlement: float,
    local_density: int,
) -> str:
    """Return the prior key for a cell given its features."""
    near = dist_to_settlement <= 5 or local_density >= 2

    if terrain_code == TerrainCode.OCEAN:
        return "ocean"
    if terrain_code == TerrainCode.MOUNTAIN:
        return "mountain"
    if terrain_code == TerrainCode.FOREST:
        return "forest_near_settle" if near else "forest_isolated"
    if terrain_code == TerrainCode.SETTLEMENT:
        return "settlement_coastal" if is_coastal else "settlement_inland"
    if terrain_code == TerrainCode.PORT:
        return "port"
    if terrain_code == TerrainCode.RUIN:
        if is_coastal:
            return "ruin_coastal"
        return "ruin_near_settle" if near else "ruin_isolated"
    # Plains (11), Empty (0) — fall through
    if is_coastal:
        return "plains_coastal_near" if near else "plains_coastal_far"
    return "plains_near" if near else "plains_far"


class RuleBasedPredictor:
    """Generates H×W×6 probability tensors from deterministic features + observations.

    Usage:
        predictor = RuleBasedPredictor(world, store)
        probs = predictor.predict(seed_index)  # np.ndarray [H, W, 6]
    """

    def __init__(self, world: WorldState, store: ObservationStore):
        self._world = world
        self._store = store

    def predict(self, seed_index: int, params: HiddenParams | None = None) -> np.ndarray:
        """Build a [H, W, 6] probability array for one seed.

        Steps:
        1. Build per-cell Dirichlet alpha from terrain features.
        2. Update alpha with post-simulation observations (Bayesian counts).
        3. Apply inferred hidden-parameter adjustments (cross-seed learning).
        4. Convert to probabilities, apply floor, renormalise.

        Parameters
        ----------
        seed_index : which seed to predict for
        params     : pre-computed HiddenParams (pass to avoid re-inferring for each seed)
        """
        sw = self._world.seed(seed_index)
        round_id = self._world.round_id

        # Step 1: prior
        alpha = self._build_prior(sw)

        # Step 2: direct observations for this seed
        observations = self._store.load_for_seed(round_id, seed_index)
        alpha = self._update_with_observations(alpha, observations, sw)

        # Step 3: cross-seed hidden-parameter adjustment
        if params is None:
            params = infer_params(self._world, self._store)
        alpha = apply_params_to_prior(alpha, sw, params)

        # Step 4: normalise, floor, renormalise
        probs = alpha / alpha.sum(axis=-1, keepdims=True)
        probs = apply_probability_floor(probs, floor=PROB_FLOOR)
        return probs

    def predict_all_seeds(self) -> list[np.ndarray]:
        """Predict all seeds, inferring hidden params once and sharing across seeds."""
        params = infer_params(self._world, self._store)
        logger.info("Predicting all %d seeds with shared hidden params", self._world.seeds_count)
        return [self.predict(i, params=params) for i in range(self._world.seeds_count)]

    def predict_with_surrogate(
        self,
        seed_index: int,
        params: HiddenParams | None = None,
        n_runs: int = 200,
        blend: float | None = None,
    ) -> np.ndarray:
        """Predict using the local surrogate simulator, blended with rule-based prior.

        The surrogate captures spatial dynamics (expansion competition, cascading
        collapses, trade networks) that flat per-cell rules cannot.  It runs the
        full Norse simulation n_runs times with the inferred hidden parameters and
        returns the empirical distribution.

        The result is blended with the rule-based prediction:
            final = (1 - w) * rule_based + w * surrogate

        The blend weight w is determined automatically from parameter confidence
        (higher confidence → more weight on surrogate) unless overridden.

        Parameters
        ----------
        seed_index : which seed to predict
        params     : pre-inferred HiddenParams; computed fresh if None
        n_runs     : how many simulation runs (more = more accurate, but slower)
        blend      : manual blend weight [0,1]; None = auto from confidence
        """
        from ..sim.simulator import Simulator, SimParams

        if params is None:
            params = infer_params(self._world, self._store)

        sw = self._world.seed(seed_index)

        # --- Rule-based prediction (prior + obs + param adjustments) ---
        rule_pred = self.predict(seed_index, params=params)

        # --- Surrogate prediction ---
        sim_params = SimParams.from_hidden_params(params)
        simulator = Simulator(sim_params, sw.height, sw.width)
        logger.info(
            "Running surrogate: seed=%d  n_runs=%d  winter=%.2f  expansion=%.2f",
            seed_index, n_runs, params.winter_severity, params.expansion_rate,
        )
        surrogate_pred = simulator.simulate_distribution(
            sw.terrain.tolist(),
            sw.seeds_settlements(seed_index) if hasattr(sw, "seeds_settlements") else
            _seed_world_to_settlements(sw),
            n_runs=n_runs,
        )

        # --- Blend ---
        if blend is None:
            # Auto: average confidence across the most important params
            import math
            n_half = 5.0
            confs = [
                1 - math.exp(-params.n_survival / n_half),
                1 - math.exp(-params.n_expansion / n_half),
                1 - math.exp(-params.n_ruin / n_half),
            ]
            mean_conf = sum(confs) / len(confs)
            # Even with 0 observations the surrogate with default params is useful,
            # so give it at least 40% weight; max out at 85% with high confidence.
            blend = 0.40 + mean_conf * 0.45

        logger.info("Blending rule=%.2f  surrogate=%.2f", 1 - blend, blend)
        blended = (1.0 - blend) * rule_pred + blend * surrogate_pred
        return apply_probability_floor(blended)

    def predict_all_seeds_with_surrogate(
        self,
        n_runs: int = 200,
    ) -> list[np.ndarray]:
        """Predict all seeds with the surrogate, inferring params once."""
        params = infer_params(self._world, self._store)
        logger.info(
            "Surrogate prediction for all %d seeds (%d runs each)",
            self._world.seeds_count, n_runs,
        )
        return [
            self.predict_with_surrogate(i, params=params, n_runs=n_runs)
            for i in range(self._world.seeds_count)
        ]


    def _build_prior(self, sw: SeedWorld) -> np.ndarray:
        """Return [H, W, 6] Dirichlet alpha array from terrain features."""
        H, W = sw.height, sw.width
        alpha = np.zeros((H, W, N_CLASSES), dtype=np.float64)

        for y in range(H):
            for x in range(W):
                code = int(sw.terrain[y, x])
                is_coastal = bool(sw.coast_mask[y, x])
                dist = float(sw.settlement_distance[y, x])
                density = int(sw.local_settlement_density[y, x])
                key = _prior_key(code, is_coastal, dist, density)
                alpha[y, x] = _BASE_PRIORS[key] * _PRIOR_STRENGTH

        return alpha

    def _update_with_observations(
        self,
        alpha: np.ndarray,
        observations: list[Observation],
        sw: SeedWorld,
    ) -> np.ndarray:
        """Increment alpha by 1 for each observed post-simulation cell outcome.

        Static cells are skipped (we know their class exactly from the mask).
        Multiple overlapping observations of the same cell accumulate naturally.
        """
        for obs in observations:
            for wx, wy, code in obs.iter_cells():
                if wx >= sw.width or wy >= sw.height:
                    continue
                if sw.static_mask[wy, wx]:
                    continue
                cls = TERRAIN_TO_CLASS.get(code)
                if cls is None:
                    continue
                alpha[wy, wx, cls] += 1.0
        return alpha


def _seed_world_to_settlements(sw) -> list:
    """Convert SeedWorld settlement positions back to a Settlement-like list."""
    from ..api.client import Settlement
    port_set = set(sw.port_positions)
    return [
        Settlement(x=x, y=y, has_port=(x, y) in port_set, alive=True)
        for x, y in sw.settlement_positions
    ]


def apply_probability_floor(probs: np.ndarray, floor: float = PROB_FLOOR) -> np.ndarray:
    """Apply minimum probability floor and renormalise.

    From scoring docs: never assign 0.0 — KL divergence becomes infinite.
    Recommended floor: 0.01 per class.
    """
    probs = np.maximum(probs, floor)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    return probs


def prediction_to_list(probs: np.ndarray) -> list:
    """Convert [H, W, 6] numpy array to nested Python list for JSON serialisation."""
    return probs.tolist()
