"""Train a convolutional segmentation model for full-map prediction.

Architecture: small fully-convolutional network (no pooling) that takes
the entire 40x40 map as an image and predicts per-cell class probabilities.

Input channels (13 total):
    0-5:  initial prior probabilities (from raw terrain codes)
    6-11: empirical_counts normalised (from queries executed this round)
    12:   observed_fraction per cell (0=never queried, 1=queried once+)

Output: [H, W, 6] softmax probabilities

For rounds without stored queries, channels 6-12 are all zeros — the CNN
learns to handle both the "before queries" and "after queries" cases.
"""
from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..utils.logging import get_logger

log = get_logger(__name__)

# ── Architecture ────────────────────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _check_torch():
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required: pip install torch")


class MapSegCNN(nn.Module):
    """Small fully-convolutional segmentation network for 40x40 maps."""

    IN_CHANNELS = 13   # 6 prior + 6 empirical_counts + 1 observed_mask
    OUT_CHANNELS = 6

    def __init__(self, hidden: int = 64):
        super().__init__()
        ic = self.IN_CHANNELS
        self.net = nn.Sequential(
            nn.Conv2d(ic,       hidden,   3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden,   hidden,   3, padding=1), nn.ReLU(),
            nn.Dropout2d(0.25),
            nn.Conv2d(hidden,   hidden*2, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(hidden*2, hidden*2, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Dropout2d(0.25),
            nn.Conv2d(hidden*2, hidden,   3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden,   self.OUT_CHANNELS, 1),
        )

    def forward(self, x):
        return self.net(x)

    def predict_proba(self, x_np: np.ndarray) -> np.ndarray:
        """Numpy convenience wrapper used by infer.py.

        Args:
            x_np: [H, W, 13] float32 input
        Returns:
            [H, W, 6] float32 probability map
        """
        _check_torch()
        self.eval()
        with torch.no_grad():
            t = torch.tensor(x_np.transpose(2, 0, 1)[None], dtype=torch.float32)
            out = F.softmax(self(t), dim=1)[0].numpy()  # [6, H, W]
        return out.transpose(1, 2, 0)  # [H, W, 6]


# ── Dataset builder ─────────────────────────────────────────────────────────

def _build_cnn_sample(
    raw_grid: np.ndarray,
    empirical_counts: np.ndarray,
    observed_count: np.ndarray,
) -> np.ndarray:
    """Build [H, W, 13] input tensor for one seed.

    Args:
        raw_grid:        [H, W] int16 raw terrain codes
        empirical_counts:[H, W, 6] float64 query observation counts
        observed_count:  [H, W] int32 number of times each cell queried
    """
    from ..map.encoding import build_prior_map

    H, W = raw_grid.shape
    prior = build_prior_map(raw_grid).astype(np.float32)         # [H, W, 6]

    # Normalise empirical counts to sum-to-1 per cell; zero if unobserved
    total = empirical_counts.sum(axis=-1, keepdims=True)         # [H, W, 1]
    counts_norm = np.where(total > 0, empirical_counts / total, 0.0).astype(np.float32)

    # Observed fraction: clip to [0,1]
    obs_frac = np.clip(observed_count.astype(np.float32) / max(1, observed_count.max()), 0, 1)

    return np.concatenate([prior, counts_norm, obs_frac[..., None]], axis=-1)  # [H,W,13]


def build_cnn_dataset(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load all completed rounds and build CNN training tensors.

    Returns:
        X:      [N, H, W, 13] float32 inputs
        Y:      [N, H, W]     int64  class labels
        groups: [N]           round index for GroupKFold
    """
    from ..online.round_state import RoundState
    from ..utils.io import load_json

    data_dir = Path(data_dir)
    X_list, Y_list, groups_list = [], [], []
    round_idx = 0

    for round_dir in sorted((data_dir / "rounds").iterdir()):
        analysis_dir = round_dir / "analysis"
        detail_file = round_dir / "detail.json"
        if not analysis_dir.exists() or not detail_file.exists():
            continue

        detail = load_json(detail_file)
        state = RoundState.from_detail(detail)

        # Replay stored queries if available
        query_dir = round_dir / "queries"
        if query_dir.exists():
            for qf in sorted(query_dir.glob("*.json")):
                try:
                    q = load_json(qf)
                    result = q.get("result", q)  # some files store result at top level
                    meta = q.get("metadata", {})
                    if "grid" not in result:
                        continue
                    state.update_from_query(
                        seed_idx=meta.get("seed_idx", meta.get("seed_index", 0)),
                        viewport_x=meta.get("x", meta.get("viewport_x", 0)),
                        viewport_y=meta.get("y", meta.get("viewport_y", 0)),
                        viewport_w=meta.get("w", meta.get("viewport_w", 15)),
                        viewport_h=meta.get("h", meta.get("viewport_h", 15)),
                        grid_result=result["grid"],
                    )
                except Exception:
                    continue

        any_seed = False
        for seed_idx in range(state.seeds_count):
            af = analysis_dir / f"seed_{seed_idx}.json"
            if not af.exists():
                continue
            analysis = load_json(af)
            gt = analysis.get("ground_truth")
            if gt is None:
                continue

            gt_arr = np.array(gt, dtype=np.int64)
            if gt_arr.ndim == 3:
                # one-hot [H, W, 6] → class indices
                gt_arr = gt_arr.argmax(axis=-1)

            x = _build_cnn_sample(
                state.raw_grids[seed_idx],
                state.empirical_counts[seed_idx],
                state.observed_count[seed_idx],
            )
            X_list.append(x)
            Y_list.append(gt_arr)
            groups_list.append(round_idx)
            any_seed = True

        if any_seed:
            log.info(f"Loaded round {round_dir.name} (idx={round_idx})")
            round_idx += 1

    X = np.stack(X_list).astype(np.float32)
    Y = np.stack(Y_list).astype(np.int64)
    groups = np.array(groups_list, dtype=np.int64)
    log.info(f"CNN dataset: {len(X)} samples from {round_idx} rounds, shape {X.shape}")
    return X, Y, groups


# ── Training ─────────────────────────────────────────────────────────────────

def _augment(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Random rotation + flip augmentation. x:[H,W,C], y:[H,W]"""
    k = np.random.randint(4)
    x = np.rot90(x, k, axes=(0, 1)).copy()
    y = np.rot90(y, k).copy()
    if np.random.rand() > 0.5:
        x = np.flip(x, axis=1).copy()
        y = np.flip(y, axis=1).copy()
    return x, y


def train_cnn(
    data_dir: Path,
    models_dir: Path,
    n_augment: int = 7,
    epochs: int = 120,
    hidden: int = 64,
    lr: float = 5e-4,
) -> MapSegCNN:
    """Train CNN and save to models_dir/cnn.pt.

    Returns the trained model.
    """
    _check_torch()
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.model_selection import GroupKFold

    data_dir = Path(data_dir)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    X, Y, groups = build_cnn_dataset(data_dir)
    n_rounds = len(np.unique(groups))

    # ── Cross-val to measure OOF KL ──────────────────────────────────────
    gkf = GroupKFold(n_splits=min(5, n_rounds))
    oof_kls = []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, Y[:, 0, 0], groups)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        Y_tr, Y_val = Y[tr_idx], Y[val_idx]

        # Augment
        aug_X = list(X_tr); aug_Y = list(Y_tr)
        for xi, yi in zip(X_tr, Y_tr):
            for _ in range(n_augment):
                ax, ay = _augment(xi, yi)
                aug_X.append(ax); aug_Y.append(ay)
        aug_X = np.stack(aug_X); aug_Y = np.stack(aug_Y)

        Xtr_t = torch.tensor(aug_X.transpose(0, 3, 1, 2), dtype=torch.float32)
        Ytr_t = torch.tensor(aug_Y, dtype=torch.long)
        Xval_t = torch.tensor(X_val.transpose(0, 3, 1, 2), dtype=torch.float32)

        counts = np.bincount(aug_Y.ravel(), minlength=6).astype(np.float32)
        counts = np.where(counts == 0, 1, counts)
        weights = torch.tensor(counts.sum() / (6 * counts), dtype=torch.float32)

        model = MapSegCNN(hidden=hidden)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        loader = DataLoader(TensorDataset(Xtr_t, Ytr_t), batch_size=16, shuffle=True)

        for epoch in range(epochs):
            model.train()
            for xb, yb in loader:
                loss = F.cross_entropy(model(xb), yb, weight=weights)
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()

        model.eval()
        with torch.no_grad():
            probs = F.softmax(model(Xval_t), dim=1).numpy().transpose(0, 2, 3, 1)

        kl_vals = []
        for n in range(len(val_idx)):
            p_onehot = np.eye(6)[Y_val[n].ravel()]  # [H*W, 6]
            q = np.clip(probs[n].reshape(-1, 6), 1e-7, 1.0)
            kl = np.where(p_onehot > 0, p_onehot * np.log(p_onehot / q), 0).sum(axis=1).mean()
            kl_vals.append(float(kl))

        fold_kl = float(np.mean(kl_vals))
        oof_kls.append(fold_kl)
        log.info(f"CNN fold {fold+1}/{gkf.n_splits}: KL={fold_kl:.4f}")

    oof_kl = float(np.mean(oof_kls))
    log.info(f"CNN OOF KL={oof_kl:.4f}")

    # ── Train final model on all data ─────────────────────────────────────
    aug_X = list(X); aug_Y = list(Y)
    for xi, yi in zip(X, Y):
        for _ in range(n_augment):
            ax, ay = _augment(xi, yi)
            aug_X.append(ax); aug_Y.append(ay)
    aug_X = np.stack(aug_X); aug_Y = np.stack(aug_Y)

    Xall_t = torch.tensor(aug_X.transpose(0, 3, 1, 2), dtype=torch.float32)
    Yall_t = torch.tensor(aug_Y, dtype=torch.long)

    counts = np.bincount(aug_Y.ravel(), minlength=6).astype(np.float32)
    counts = np.where(counts == 0, 1, counts)
    weights = torch.tensor(counts.sum() / (6 * counts), dtype=torch.float32)

    final_model = MapSegCNN(hidden=hidden)
    opt = torch.optim.Adam(final_model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(TensorDataset(Xall_t, Yall_t), batch_size=16, shuffle=True)

    for epoch in range(epochs):
        final_model.train()
        for xb, yb in loader:
            loss = F.cross_entropy(final_model(xb), yb, weight=weights)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

    # Save
    cnn_path = models_dir / "cnn.pt"
    torch.save({
        "state_dict": final_model.state_dict(),
        "hidden": hidden,
        "oof_kl": oof_kl,
        "n_rounds": n_rounds,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }, cnn_path)
    log.info(f"CNN saved to {cnn_path}  (OOF KL={oof_kl:.4f})")
    return final_model


def load_cnn(models_dir: Path) -> MapSegCNN | None:
    """Load CNN from models_dir/cnn.pt. Returns None if not found."""
    _check_torch()
    import torch

    cnn_path = Path(models_dir) / "cnn.pt"
    if not cnn_path.exists():
        return None
    try:
        ckpt = torch.load(cnn_path, map_location="cpu", weights_only=True)
        model = MapSegCNN(hidden=ckpt.get("hidden", 64))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        log.info(
            f"Loaded CNN from {cnn_path} "
            f"(OOF KL={ckpt.get('oof_kl','?')}, rounds={ckpt.get('n_rounds','?')})"
        )
        return model
    except Exception as e:
        log.warning(f"Failed to load CNN: {e}")
        return None
