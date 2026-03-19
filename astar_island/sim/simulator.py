"""Local surrogate simulator for the Norse civilisation world.

This is a simplified but mechanically faithful reimplementation of the simulation
described in the API docs.  It captures the five phases each year:

    Growth → Conflict → Trade → Winter → Environment

Purpose
-------
Run this hundreds of times offline with different random seeds to generate a
*probability distribution* over final terrain states.  When we infer hidden
parameters from real observations, we feed those into this simulator to get
a much better prediction than rule-based priors alone.

The key insight: the surrogate captures *spatial interactions* (one settlement
collapsing weakens its neighbours, ports trade with nearby ports, etc.) that
flat per-cell rules cannot represent.

Design choices
--------------
- Parameterised with the same 6 HiddenParam dimensions so they map directly.
- Numpy-vectorised for the environment phase (the bottleneck on large maps).
- Fully deterministic given rng_seed — reproducible for debugging.
- No faction alliances: every initial settlement is its own faction.
  Newly founded settlements inherit the parent's faction.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from ..world.terrain import TerrainCode, TERRAIN_TO_CLASS
from ..models.predictor import apply_probability_floor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class SimParams:
    """The 6 hidden parameters + fixed simulation constants.

    The six hidden fields map 1-to-1 to HiddenParams from param_inference.py.
    """
    # Hidden (vary per round, inferred from observations)
    winter_severity:  float = 0.5   # 0=mild, 1=brutal
    expansion_rate:   float = 0.5   # 0=no growth, 1=aggressive colonisation
    raid_intensity:   float = 0.5   # 0=peaceful, 1=constant warfare
    ruin_recovery:    float = 0.5   # 0=ruins persist, 1=quick reclamation
    port_development: float = 0.5   # 0=no ports built, 1=many ports
    forest_growth:    float = 0.5   # 0=no spread, 1=rapid forest reclaim

    # Fixed physics constants (calibrated to produce realistic dynamics)
    # Food production uses 8-directional neighbors, so typical plains settlement
    # produces ~12/year. Winter mean loss scales with severity so that:
    #   mild  (0.05) → ~95% survival over 50 years
    #   medium(0.50) → ~60% survival
    #   harsh (0.95) → ~15% survival
    food_from_forest:      float = 3.0   # food per adjacent (8-dir) forest cell per year
    food_from_plains:      float = 1.5   # food per adjacent (8-dir) plains/empty cell per year
    food_consumption:      float = 0.15  # food consumed per population unit per year
    winter_food_loss:      float = 4.0   # food deducted each winter (for prosperity tracking)
    pop_growth_rate:       float = 0.18  # annual fractional population growth when prosperous
    pop_max:               float = 60.0  # maximum population per settlement
    expansion_threshold:   float = 20.0  # population needed to consider expanding
    expansion_range:       int   = 5     # max Manhattan distance for new settlement
    raid_range_base:       int   = 6     # base raiding range (cells)
    raid_range_per_ship:   int   = 4     # extra range per longship
    ruin_recovery_range:   int   = 6     # max range from sponsor settlement for recovery
    forest_min_dist:       int   = 7     # min distance from all settlements for forest spread
    port_wealth_threshold: float = 12.0  # wealth needed before port can develop
    trade_range:           int   = 14    # max port-to-port trade range

    @classmethod
    def from_hidden_params(cls, hp) -> "SimParams":
        """Create SimParams from a HiddenParams inference result."""
        return cls(
            winter_severity=hp.winter_severity,
            expansion_rate=hp.expansion_rate,
            raid_intensity=hp.raid_intensity,
            ruin_recovery=hp.ruin_recovery,
            port_development=hp.port_development,
            forest_growth=hp.forest_growth,
        )


# ---------------------------------------------------------------------------
# Settlement state
# ---------------------------------------------------------------------------

@dataclass
class SimSettlement:
    x: int
    y: int
    population: float
    food: float
    wealth: float
    has_port: bool
    longships: int
    owner_id: int
    alive: bool = True


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class Simulator:
    """Stochastic Norse civilisation simulator."""

    YEARS = 50

    def __init__(self, params: SimParams, height: int, width: int):
        self.p = params
        self.H = height
        self.W = width
        # Row/col index grids for vectorised distance calculations
        self._ys = np.arange(height, dtype=np.float32).reshape(-1, 1)
        self._xs = np.arange(width,  dtype=np.float32).reshape(1, -1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate(
        self,
        initial_grid: list[list[int]],
        initial_settlements: list,
        rng_seed: int = 0,
    ) -> np.ndarray:
        """Run one 50-year simulation.

        Returns
        -------
        final_grid : np.ndarray [H, W] of terrain codes after 50 years
        """
        rng = np.random.default_rng(rng_seed)
        grid = np.array(initial_grid, dtype=np.int32)
        settlements = self._init_settlements(initial_settlements)

        for _ in range(self.YEARS):
            self._phase_growth(settlements, grid, rng)
            self._phase_conflict(settlements, rng)
            self._phase_trade(settlements)
            self._phase_winter(settlements, rng)
            self._phase_environment(settlements, grid, rng)
            self._sync_grid(settlements, grid)

        return grid

    def simulate_distribution(
        self,
        initial_grid: list[list[int]],
        initial_settlements: list,
        n_runs: int = 200,
    ) -> np.ndarray:
        """Run n_runs stochastic simulations and return a [H, W, 6] probability array.

        Each run uses a different rng_seed so we get independent samples from
        the distribution.  More runs → lower variance in the probability estimate.
        200 runs is typically enough for stable predictions on a 40×40 map.
        """
        counts = np.zeros((self.H, self.W, 6), dtype=np.float64)
        for run in range(n_runs):
            final_grid = self.simulate(initial_grid, initial_settlements, rng_seed=run)
            # Vectorised accumulation: convert all terrain codes to class indices at once
            class_grid = np.zeros((self.H, self.W), dtype=np.int32)
            for code, cls in TERRAIN_TO_CLASS.items():
                class_grid[final_grid == code] = cls
            for cls in range(6):
                counts[:, :, cls] += (class_grid == cls)

        probs = counts / n_runs
        return apply_probability_floor(probs)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_settlements(self, initial_settlements: list) -> list[SimSettlement]:
        out = []
        for i, s in enumerate(initial_settlements):
            if not s.alive:
                continue
            out.append(SimSettlement(
                x=s.x,
                y=s.y,
                population=10.0,
                food=40.0,
                wealth=12.0,
                has_port=s.has_port,
                longships=2 if s.has_port else 0,
                owner_id=i,
            ))
        return out

    # ------------------------------------------------------------------
    # Phase 1: Growth
    # ------------------------------------------------------------------

    def _phase_growth(self, settlements: list, grid: np.ndarray, rng: np.random.Generator) -> None:
        p = self.p
        for s in settlements:
            if not s.alive:
                continue

            # Food production from adjacent terrain
            s.food += self._food_production(s.x, s.y, grid)

            # Consumption
            s.food -= s.population * p.food_consumption

            # Population grows when food is positive
            if s.food > 0:
                # Exponential growth proportional to food surplus per capita
                surplus_per_cap = min(1.0, s.food / (s.population * 5.0 + 1.0))
                s.population = min(p.pop_max, s.population * (1.0 + p.pop_growth_rate * surplus_per_cap))
                s.wealth += surplus_per_cap * 2.0

            # Port development (coastal + wealthy enough)
            if not s.has_port and s.wealth >= p.port_wealth_threshold:
                if self._is_coastal(s.x, s.y, grid):
                    if rng.random() < p.port_development * 0.18:
                        s.has_port = True
                        s.longships += 1

            # Expansion when large enough and not starving
            if s.population >= p.expansion_threshold and s.food > s.population:
                if rng.random() < p.expansion_rate * 0.15:
                    self._try_expand(s, settlements, grid, rng)

    def _food_production(self, x: int, y: int, grid: np.ndarray) -> float:
        """Sum food from all 8 neighbors (including diagonals)."""
        total = 0.0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.H and 0 <= nx < self.W:
                    tc = int(grid[ny, nx])
                    if tc == TerrainCode.FOREST:
                        total += self.p.food_from_forest
                    elif tc in (TerrainCode.PLAINS, TerrainCode.EMPTY):
                        total += self.p.food_from_plains
        return total

    def _is_coastal(self, x: int, y: int, grid: np.ndarray) -> bool:
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.H and 0 <= nx < self.W:
                if grid[ny, nx] == TerrainCode.OCEAN:
                    return True
        return False

    def _try_expand(
        self,
        parent: SimSettlement,
        settlements: list,
        grid: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        occupied = {(s.x, s.y) for s in settlements if s.alive}
        r = self.p.expansion_range
        candidates: list[tuple[int, int, int]] = []

        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                nx, ny = parent.x + dx, parent.y + dy
                if not (0 <= ny < self.H and 0 <= nx < self.W):
                    continue
                if (nx, ny) in occupied:
                    continue
                tc = int(grid[ny, nx])
                if tc in (TerrainCode.PLAINS, TerrainCode.EMPTY, TerrainCode.FOREST):
                    candidates.append((abs(dx) + abs(dy), nx, ny))

        if not candidates:
            return

        candidates.sort()
        _, nx, ny = candidates[0]

        new_s = SimSettlement(
            x=nx, y=ny,
            population=3.0,
            food=parent.food * 0.25,
            wealth=parent.wealth * 0.25,
            has_port=False,
            longships=0,
            owner_id=parent.owner_id,
        )
        parent.food      *= 0.88   # smaller cost so repeated expansion stays viable
        parent.wealth    *= 0.88
        parent.population *= 0.88
        settlements.append(new_s)
        grid[ny, nx] = TerrainCode.SETTLEMENT

    # ------------------------------------------------------------------
    # Phase 2: Conflict
    # ------------------------------------------------------------------

    def _phase_conflict(self, settlements: list, rng: np.random.Generator) -> None:
        alive = [s for s in settlements if s.alive]
        factions = {s.owner_id for s in alive}
        if len(factions) < 2:
            return  # No enemies → no raids

        for s in alive:
            # Desperate settlements raid more; well-fed ones rarely raid
            food_ratio = s.food / max(1.0, s.population * 5.0)
            raid_prob = self.p.raid_intensity * 0.35 * max(0.1, 1.2 - food_ratio)
            if rng.random() > raid_prob:
                continue

            raid_range = self.p.raid_range_base + s.longships * self.p.raid_range_per_ship
            targets = [
                t for t in alive
                if t.owner_id != s.owner_id
                and abs(t.x - s.x) + abs(t.y - s.y) <= raid_range
            ]
            if not targets:
                continue

            target = min(targets, key=lambda t: abs(t.x - s.x) + abs(t.y - s.y))

            attack  = s.population + s.wealth * 0.1 + s.longships * 2
            defense = target.population + target.wealth * 0.1
            win_prob = attack / (attack + defense + 1e-9)

            if rng.random() < win_prob:
                loot = min(target.wealth * 0.4, 20.0)
                target.wealth -= loot
                target.food   -= min(target.food * 0.30, 5.0)
                s.wealth      += loot * 0.7
                # Rare conquest: flip allegiance
                if rng.random() < 0.12:
                    target.owner_id = s.owner_id
            else:
                # Defender repels attacker
                s.food -= min(s.food * 0.20, 3.0)

    # ------------------------------------------------------------------
    # Phase 3: Trade
    # ------------------------------------------------------------------

    def _phase_trade(self, settlements: list) -> None:
        ports = [s for s in settlements if s.alive and s.has_port]
        for i, a in enumerate(ports):
            for b in ports[i + 1:]:
                dist = abs(a.x - b.x) + abs(a.y - b.y)
                if dist > self.p.trade_range:
                    continue
                # Allied or same faction: trade
                if a.owner_id != b.owner_id:
                    continue
                trade = min(a.wealth, b.wealth, 6.0) * 0.12
                a.food   += trade
                b.food   += trade
                a.wealth += trade * 0.4
                b.wealth += trade * 0.4

    # ------------------------------------------------------------------
    # Phase 4: Winter
    # ------------------------------------------------------------------

    def _phase_winter(self, settlements: list, rng: np.random.Generator) -> None:
        """Winter phase: food depletion + calibrated stochastic kill.

        Kill probability is calibrated analytically so that:
            severity=0.05 → ~95% survival over 50 years
            severity=0.50 → ~60% survival over 50 years
            severity=0.95 → ~10% survival over 50 years

        Formula: base_kill = max(0, -0.028 + 0.076 * severity)
            At 0.05: 0%  → 0.95^0/year → 100% survival
            At 0.50: 1%  → 0.99^50 ≈ 61%
            At 0.95: 4.4%→ 0.956^50 ≈ 10%

        Settlement health (food vs population) scales the kill up for weak settlements.
        """
        base_kill = max(0.0, -0.028 + 0.076 * self.p.winter_severity)

        for s in settlements:
            if not s.alive:
                continue

            # Food loss reduces prosperity (affects expansion/port thresholds)
            s.food  -= self.p.winter_food_loss * (0.5 + self.p.winter_severity * 0.5)
            s.wealth -= 0.5

            # Guaranteed collapse if food runs completely dry
            if s.food < 0:
                s.alive = False
                continue

            # Probability kill: weak settlements (low food/pop ratio) are more vulnerable
            health = min(1.0, s.food / (s.population * 6.0 + 1.0))
            kill_p = base_kill * (1.8 - health)   # [base_kill * 0.8, base_kill * 1.8]
            if rng.random() < kill_p:
                s.alive = False

    # ------------------------------------------------------------------
    # Phase 5: Environment
    # ------------------------------------------------------------------

    def _phase_environment(
        self,
        settlements: list,
        grid: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        alive = [s for s in settlements if s.alive]
        dist_map = self._min_dist_to_settlements(alive)   # [H, W] Manhattan distance

        # --- Ruin recovery ---
        # Each ruin near a thriving settlement may be rebuilt
        ruin_mask = grid == TerrainCode.RUIN
        near_mask = dist_map <= self.p.ruin_recovery_range
        recoverable = ruin_mask & near_mask
        ry_coords, rx_coords = np.where(recoverable)

        for ry, rx in zip(ry_coords.tolist(), rx_coords.tolist()):
            if rng.random() >= self.p.ruin_recovery * 0.15:
                continue
            # Find nearest alive settlement to inherit resources from
            sponsors = sorted(alive, key=lambda s: abs(s.x - rx) + abs(s.y - ry))
            if not sponsors:
                continue
            best = sponsors[0]
            new_s = SimSettlement(
                x=rx, y=ry,
                population=3.0,
                food=best.food * 0.20,
                wealth=best.wealth * 0.20,
                has_port=self._is_coastal(rx, ry, grid),
                longships=0,
                owner_id=best.owner_id,
            )
            settlements.append(new_s)
            best.food   *= 0.82
            best.wealth *= 0.82

        # --- Forest spread ---
        # Empty / plains land far from all settlements slowly becomes forest
        forest_candidates = (
            ((grid == TerrainCode.PLAINS) | (grid == TerrainCode.EMPTY))
            & (dist_map > self.p.forest_min_dist)
        )
        fy_coords, fx_coords = np.where(forest_candidates)
        if len(fy_coords) > 0:
            roll = rng.random(len(fy_coords))
            threshold = self.p.forest_growth * 0.04
            for i, (fy, fx) in enumerate(zip(fy_coords.tolist(), fx_coords.tolist())):
                if roll[i] < threshold:
                    grid[fy, fx] = TerrainCode.FOREST

    # ------------------------------------------------------------------
    # Grid sync
    # ------------------------------------------------------------------

    def _sync_grid(self, settlements: list, grid: np.ndarray) -> None:
        """Write settlement states back to the terrain grid."""
        for s in settlements:
            if s.alive:
                grid[s.y, s.x] = TerrainCode.PORT if s.has_port else TerrainCode.SETTLEMENT
            else:
                # Only mark as ruin if the cell currently shows a settlement/port
                # (it might have been overwritten by a newly founded settlement)
                if int(grid[s.y, s.x]) in (TerrainCode.SETTLEMENT, TerrainCode.PORT):
                    grid[s.y, s.x] = TerrainCode.RUIN

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _min_dist_to_settlements(self, alive: list) -> np.ndarray:
        """[H, W] Manhattan distance to the nearest alive settlement.

        Uses numpy broadcasting for speed — O(n_alive) passes, each O(H*W).
        Returns array of 999 where no settlements are alive.
        """
        dist = np.full((self.H, self.W), 999.0, dtype=np.float32)
        for s in alive:
            d = np.abs(self._ys - s.y) + np.abs(self._xs - s.x)
            np.minimum(dist, d, out=dist)
        return dist
