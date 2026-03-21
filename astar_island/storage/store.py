"""Immutable round data persistence.

Layout inside data_dir/rounds/{round_id}/:
    meta.json
    detail.json
    raw_grids_seed_{k}.npz
    encoded_grids_seed_{k}.npz
    queries/
        q_0000.json ... q_XXXX.json
    submissions/
        seed_{k}.json           ← metadata + tensor path
        seed_{k}_tensor.npz     ← actual prediction tensor
    analysis/
        seed_{k}.json
"""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.io import save_json, load_json, save_npz, load_npz, ensure_dir


class RoundStore:
    """Filesystem store for a single round."""

    def __init__(self, data_dir: str | Path, round_id: str) -> None:
        self._root = Path(data_dir) / "rounds" / round_id
        ensure_dir(self._root / "queries")
        ensure_dir(self._root / "submissions")
        ensure_dir(self._root / "analysis")

    # ------------------------------------------------------------------
    # Round metadata
    # ------------------------------------------------------------------

    def save_meta(self, meta: dict) -> None:
        save_json(self._root / "meta.json", meta)

    def save_detail(self, detail: dict) -> None:
        save_json(self._root / "detail.json", detail)

    def load_meta(self) -> dict:
        return load_json(self._root / "meta.json")

    def load_detail(self) -> dict:
        return load_json(self._root / "detail.json")

    def meta_exists(self) -> bool:
        return (self._root / "meta.json").exists()

    # ------------------------------------------------------------------
    # Initial grids
    # ------------------------------------------------------------------

    def save_raw_grid(self, seed_idx: int, raw_grid: np.ndarray) -> None:
        save_npz(self._root / f"raw_grids_seed_{seed_idx}.npz", grid=raw_grid)

    def save_encoded_grid(self, seed_idx: int, encoded_grid: np.ndarray) -> None:
        save_npz(self._root / f"encoded_grids_seed_{seed_idx}.npz", grid=encoded_grid)

    def load_raw_grid(self, seed_idx: int) -> np.ndarray:
        return load_npz(self._root / f"raw_grids_seed_{seed_idx}.npz")["grid"]

    def load_encoded_grid(self, seed_idx: int) -> np.ndarray:
        return load_npz(self._root / f"encoded_grids_seed_{seed_idx}.npz")["grid"]

    # ------------------------------------------------------------------
    # Queries  (never overwrite)
    # ------------------------------------------------------------------

    def save_query(
        self,
        query_idx: int,
        result: dict,
        metadata: dict | None = None,
    ) -> Path:
        path = self._root / "queries" / f"q_{query_idx:04d}.json"
        if path.exists():
            return path  # immutable — never overwrite
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query_idx": query_idx,
            "metadata": metadata or {},
            "result": result,
        }
        save_json(path, record)
        return path

    def query_count(self) -> int:
        return len(list((self._root / "queries").glob("q_*.json")))

    def load_queries(self) -> list[dict]:
        files = sorted((self._root / "queries").glob("q_*.json"))
        return [load_json(f) for f in files]

    # ------------------------------------------------------------------
    # Submissions
    # ------------------------------------------------------------------

    def save_submission(
        self,
        seed_idx: int,
        response: dict,
        tensor: np.ndarray,
        model_version: str = "baseline",
        config_snapshot: dict | None = None,
    ) -> None:
        tensor_path = self._root / "submissions" / f"seed_{seed_idx}_tensor.npz"
        save_npz(tensor_path, tensor=tensor)

        tensor_bytes = tensor.tobytes()
        request_hash = hashlib.sha256(tensor_bytes).hexdigest()[:16]

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "seed_idx": seed_idx,
            "round_id": self._round_id_from_root(),
            "request_hash": request_hash,
            "tensor_path": str(tensor_path.relative_to(self._root)),
            "tensor_shape": list(tensor.shape),
            "api_response": response,
            "model_version": model_version,
            "config_snapshot": config_snapshot or {},
        }
        # Always overwrite submissions — last submission is the live one
        save_json(self._root / "submissions" / f"seed_{seed_idx}.json", record)

    def load_submission(self, seed_idx: int) -> dict:
        return load_json(self._root / "submissions" / f"seed_{seed_idx}.json")

    def load_submission_tensor(self, seed_idx: int) -> np.ndarray:
        return load_npz(self._root / "submissions" / f"seed_{seed_idx}_tensor.npz")["tensor"]

    # ------------------------------------------------------------------
    # Analysis (post-round ground truth)
    # ------------------------------------------------------------------

    def save_analysis(self, seed_idx: int, analysis: dict) -> None:
        save_json(self._root / "analysis" / f"seed_{seed_idx}.json", analysis)

    def load_analysis(self, seed_idx: int) -> dict:
        return load_json(self._root / "analysis" / f"seed_{seed_idx}.json")

    def analysis_exists(self, seed_idx: int) -> bool:
        return (self._root / "analysis" / f"seed_{seed_idx}.json").exists()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _round_id_from_root(self) -> str:
        return self._root.name

    @property
    def root(self) -> Path:
        return self._root
