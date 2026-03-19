"""Authenticated REST client for the Astar Island API.

Handles:
- Bearer token auth
- Typed request/response models
- Retry with exponential backoff on transient errors
- Query budget enforcement (50 queries per round, shared across all seeds)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import requests

BASE_URL = "https://api.ainm.no"
logger = logging.getLogger(__name__)


class QueryBudgetExhausted(Exception):
    pass


class AstarAPIError(Exception):
    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.body = body
        super().__init__(f"API error {status_code}: {body}")


@dataclass
class Settlement:
    x: int
    y: int
    has_port: bool
    alive: bool

    @classmethod
    def from_dict(cls, d: dict) -> "Settlement":
        return cls(x=d["x"], y=d["y"], has_port=d["has_port"], alive=d["alive"])


@dataclass
class InitialState:
    """Initial world state for one seed before any simulation."""
    grid: list[list[int]]        # [height][width] terrain codes
    settlements: list[Settlement]

    @classmethod
    def from_dict(cls, d: dict) -> "InitialState":
        return cls(
            grid=d["grid"],
            settlements=[Settlement.from_dict(s) for s in d.get("settlements", [])],
        )


@dataclass
class Round:
    id: str
    round_number: int
    status: str
    map_width: int
    map_height: int
    seeds_count: int
    initial_states: list[InitialState]

    @classmethod
    def from_dict(cls, d: dict) -> "Round":
        return cls(
            id=d["id"],
            round_number=d["round_number"],
            status=d["status"],
            map_width=d["map_width"],
            map_height=d["map_height"],
            seeds_count=d["seeds_count"],
            initial_states=[InitialState.from_dict(s) for s in d.get("initial_states", [])],
        )


@dataclass
class SimulatedSettlement:
    """Settlement data returned after running the simulation."""
    x: int
    y: int
    has_port: bool
    alive: bool
    population: int | None = None
    food: int | None = None
    wealth: int | None = None
    defense: int | None = None
    tech_level: int | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "SimulatedSettlement":
        return cls(
            x=d["x"],
            y=d["y"],
            has_port=d.get("has_port", False),
            alive=d.get("alive", True),
            population=d.get("population"),
            food=d.get("food"),
            wealth=d.get("wealth"),
            defense=d.get("defense"),
            tech_level=d.get("tech_level"),
        )


@dataclass
class Viewport:
    x: int
    y: int
    w: int
    h: int


@dataclass
class SimulateResult:
    """Result of a single simulation query."""
    grid: list[list[int]]                    # [h][w] terrain codes after 50 years
    settlements: list[SimulatedSettlement]   # settlements visible in the viewport
    viewport: Viewport
    seed_index: int
    queries_used: int                        # budget consumed so far

    @classmethod
    def from_dict(cls, d: dict, seed_index: int, queries_used: int) -> "SimulateResult":
        vp = d["viewport"]
        return cls(
            grid=d["grid"],
            settlements=[SimulatedSettlement.from_dict(s) for s in d.get("settlements", [])],
            viewport=Viewport(x=vp["x"], y=vp["y"], w=vp["w"], h=vp["h"]),
            seed_index=seed_index,
            queries_used=queries_used,
        )


class AstarClient:
    """Authenticated client for the Astar Island API."""

    def __init__(self, token: str, budget: int = 50, max_retries: int = 3):
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {token}"
        self._budget_total = budget
        self._budget_used = 0
        self._max_retries = max_retries

    @property
    def budget_used(self) -> int:
        return self._budget_used

    @property
    def budget_remaining(self) -> int:
        return self._budget_total - self._budget_used

    def _request(self, method: str, path: str, **kwargs) -> Any:
        url = f"{BASE_URL}{path}"
        for attempt in range(self._max_retries):
            try:
                resp = self.session.request(method, url, timeout=30, **kwargs)
                if resp.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning("Rate limited; retrying in %ds", wait)
                    time.sleep(wait)
                    continue
                if resp.status_code >= 500:
                    wait = 2 ** attempt
                    logger.warning("Server error %d; retrying in %ds", resp.status_code, wait)
                    time.sleep(wait)
                    continue
                if not resp.ok:
                    raise AstarAPIError(resp.status_code, resp.text)
                return resp.json()
            except requests.RequestException as e:
                if attempt == self._max_retries - 1:
                    raise
                wait = 2 ** attempt
                logger.warning("Request failed (%s); retrying in %ds", e, wait)
                time.sleep(wait)
        raise AstarAPIError(0, "Max retries exceeded")

    def get_rounds(self) -> list[dict]:
        """Return all rounds (active, scoring, completed)."""
        return self._request("GET", "/astar-island/rounds")

    def get_active_round_id(self) -> str | None:
        """Return the ID of the currently active round, or None."""
        rounds = self.get_rounds()
        for r in rounds:
            if r.get("status") == "active":
                return r["id"]
        return None

    def get_round_detail(self, round_id: str) -> Round:
        """Fetch full round details including initial states for all seeds."""
        data = self._request("GET", f"/astar-island/rounds/{round_id}")
        round_obj = Round.from_dict(data)
        logger.info(
            "Round %d: %dx%d map, %d seeds, status=%s",
            round_obj.round_number,
            round_obj.map_width,
            round_obj.map_height,
            round_obj.seeds_count,
            round_obj.status,
        )
        return round_obj

    def simulate(
        self,
        round_id: str,
        seed_index: int,
        viewport_x: int,
        viewport_y: int,
        viewport_w: int,
        viewport_h: int,
    ) -> SimulateResult:
        """Run one stochastic simulation and return viewport observations.

        Consumes 1 query from the shared budget.
        Raises QueryBudgetExhausted if no budget remains.
        """
        if self._budget_used >= self._budget_total:
            raise QueryBudgetExhausted(
                f"Query budget exhausted ({self._budget_used}/{self._budget_total})"
            )

        # Clamp viewport dimensions to allowed range
        viewport_w = max(5, min(15, viewport_w))
        viewport_h = max(5, min(15, viewport_h))

        payload = {
            "round_id": round_id,
            "seed_index": seed_index,
            "viewport_x": viewport_x,
            "viewport_y": viewport_y,
            "viewport_w": viewport_w,
            "viewport_h": viewport_h,
        }
        logger.info(
            "Simulate seed=%d viewport=(%d,%d)+%dx%d [budget %d/%d]",
            seed_index, viewport_x, viewport_y, viewport_w, viewport_h,
            self._budget_used + 1, self._budget_total,
        )
        data = self._request("POST", "/astar-island/simulate", json=payload)
        self._budget_used += 1
        result = SimulateResult.from_dict(data, seed_index=seed_index, queries_used=self._budget_used)
        return result

    def submit(self, round_id: str, seed_index: int, prediction: list) -> dict:
        """Submit a H×W×6 probability tensor for one seed.

        prediction: nested list [height][width][6], each row sums to 1.0
        """
        _validate_prediction(prediction)
        payload = {
            "round_id": round_id,
            "seed_index": seed_index,
            "prediction": prediction,
        }
        logger.info("Submitting prediction for seed %d", seed_index)
        return self._request("POST", "/astar-island/submit", json=payload)


def _validate_prediction(prediction: list) -> None:
    """Raise ValueError if the prediction tensor is malformed."""
    import math
    height = len(prediction)
    if height == 0:
        raise ValueError("Prediction is empty")
    width = len(prediction[0])
    for y, row in enumerate(prediction):
        if len(row) != width:
            raise ValueError(f"Row {y} has width {len(row)}, expected {width}")
        for x, cell in enumerate(row):
            if len(cell) != 6:
                raise ValueError(f"Cell ({x},{y}) has {len(cell)} classes, expected 6")
            total = sum(cell)
            if not math.isclose(total, 1.0, abs_tol=1e-4):
                raise ValueError(f"Cell ({x},{y}) sums to {total:.6f}, expected 1.0")
            if any(v < 0 for v in cell):
                raise ValueError(f"Cell ({x},{y}) has negative probability")
            if any(v == 0.0 for v in cell):
                raise ValueError(f"Cell ({x},{y}) has zero probability — use minimum floor 0.01")
