"""Build cell-level training dataset from historical completed rounds."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..ingest.history import build_history
from ..utils.io import ensure_dir
from ..utils.logging import get_logger

log = get_logger(__name__)


def build_cell_dataset(data_dir: str | Path) -> pd.DataFrame:
    """Build and save parquet training dataset.

    Primary label: ground truth from analysis/ (argmax → target_class).
    Soft labels: gt_prob_{0..5} for use with KL loss.

    Saves to data_dir/processed/cell_dataset_v1.parquet.

    Returns the DataFrame.
    """
    data_dir = Path(data_dir)
    cell_df, round_df = build_history(data_dir)

    if cell_df.empty:
        log.warning("No completed rounds with analysis found — dataset is empty")
        return cell_df

    processed_dir = ensure_dir(data_dir / "processed")
    cell_path = processed_dir / "cell_dataset_v1.parquet"
    cell_df.to_parquet(cell_path, index=False)

    round_path = processed_dir / "round_features.parquet"
    round_df.to_parquet(round_path, index=False)

    log.info(
        f"Dataset saved: {len(cell_df)} cells from "
        f"{cell_df['round_id'].nunique()} rounds → {cell_path}"
    )
    return cell_df
