"""AstarClient — authenticated HTTP client for the Astar Island API.

Token: reads ASTAR_TOKEN environment variable (Bearer header).
Rate limits: simulate ≤ 5 req/s, submit ≤ 2 req/s.
"""
import os
import time
from typing import Any

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()


class AstarAPIError(Exception):
    def __init__(self, status_code: int, body: Any) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"HTTP {status_code}: {body}")


class _TokenBucket:
    """Simple token bucket for rate limiting."""

    def __init__(self, rate: float) -> None:
        self._rate = rate          # tokens per second
        self._tokens = rate
        self._last = time.monotonic()

    def consume(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        self._tokens = min(self._rate, self._tokens + elapsed * self._rate)
        self._last = now
        if self._tokens < 1.0:
            sleep_for = (1.0 - self._tokens) / self._rate
            time.sleep(sleep_for)
            self._tokens = 0.0
        else:
            self._tokens -= 1.0


class AstarClient:
    """REST client for the Astar Island API.

    Usage:
        client = AstarClient()  # reads ASTAR_TOKEN from env
        active = client.get_active_round()
        detail = client.get_round_detail(active["id"])
    """

    def __init__(
        self,
        token: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._token = token or os.environ["ASTAR_TOKEN"]
        base = (base_url or os.environ.get("BASE_URL", "https://api.ainm.no")).rstrip("/")
        self._base = base + "/astar-island"

        self._session = requests.Session()
        self._session.headers["Authorization"] = f"Bearer {self._token}"
        self._session.headers["Content-Type"] = "application/json"

        self._sim_bucket = _TokenBucket(rate=5.0)
        self._sub_bucket = _TokenBucket(rate=2.0)

    # ------------------------------------------------------------------
    # Round discovery
    # ------------------------------------------------------------------

    def list_rounds(self) -> list[dict]:
        return self._get("/rounds")

    def get_active_round(self) -> dict:
        rounds = self.list_rounds()
        active = [r for r in rounds if r.get("status") == "active"]
        if not active:
            raise RuntimeError("No active round found")
        return active[0]

    def get_round_detail(self, round_id: str) -> dict:
        return self._get(f"/rounds/{round_id}")

    def get_budget(self) -> dict:
        return self._get("/budget")

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        round_id: str,
        seed_index: int,
        x: int,
        y: int,
        w: int = 15,
        h: int = 15,
    ) -> dict:
        self._sim_bucket.consume()
        payload = {
            "round_id": round_id,
            "seed_index": seed_index,
            "viewport_x": x,
            "viewport_y": y,
            "viewport_w": w,
            "viewport_h": h,
        }
        return self._post("/simulate", payload)

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def submit(
        self,
        round_id: str,
        seed_index: int,
        prediction: np.ndarray,
    ) -> dict:
        self._sub_bucket.consume()
        payload = {
            "round_id": round_id,
            "seed_index": seed_index,
            "prediction": prediction.tolist(),
        }
        return self._post("/submit", payload)

    # ------------------------------------------------------------------
    # History & analysis
    # ------------------------------------------------------------------

    def my_rounds(self) -> list[dict]:
        return self._get("/my-rounds")

    def my_predictions(self, round_id: str) -> list[dict]:
        return self._get(f"/my-predictions/{round_id}")

    def analysis(self, round_id: str, seed_index: int) -> dict:
        return self._get(f"/analysis/{round_id}/{seed_index}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, timeout: int = 30) -> Any:
        resp = self._session.get(self._base + path, timeout=timeout)
        self._raise_for_status(resp)
        return resp.json()

    def _post(self, path: str, payload: dict, timeout: int = 120) -> Any:
        resp = self._session.post(self._base + path, json=payload, timeout=timeout)
        self._raise_for_status(resp)
        return resp.json()

    @staticmethod
    def _raise_for_status(resp: requests.Response) -> None:
        if not resp.ok:
            try:
                body = resp.json()
            except Exception:
                body = resp.text
            raise AstarAPIError(resp.status_code, body)
