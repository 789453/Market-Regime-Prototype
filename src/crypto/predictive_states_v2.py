"""Restricted geometry and distribution estimators for predictive-state ablations."""

from __future__ import annotations

import numpy as np
import torch

from .predictive_states import TASK_SIZES, CUDA_ENABLED, make_representation


def covariance_geometry(x: np.ndarray, fit: np.ndarray, group_width: int,
                        *, shrink: float = .5) -> tuple[np.ndarray, list[np.ndarray]]:
    """Whiten within each economic group, retaining equal total group scale.

    Clock coordinates are excluded by the caller. The covariance is trained only
    on pre-selection history. Eigenvalue clipping avoids a near-duplicate axis
    receiving arbitrarily large inverse variance weight.
    """
    if x.shape[1] % group_width:
        raise ValueError("geometry must contain complete economic groups")
    maps = []
    transformed = []
    for start in range(0, x.shape[1], group_width):
        part = x[:, start:start + group_width]
        history = part[fit].astype(np.float64)
        cov = np.cov(history, rowvar=False)
        diag = np.diag(np.diag(cov))
        regularized = (1 - shrink) * cov + shrink * diag
        eigenvalue, eigenvector = np.linalg.eigh(regularized)
        floor = max(np.median(np.diag(cov)) * .2, 1e-7)
        inverse_root = (eigenvector * (1 / np.sqrt(np.maximum(eigenvalue, floor)))) @ eigenvector.T
        scale = np.sqrt(np.trace(np.cov(history @ inverse_root, rowvar=False)))
        matrix = (inverse_root / max(scale, 1e-8)).astype(np.float32)
        maps.append(matrix)
        transformed.append((part @ matrix).astype(np.float32))
    return np.concatenate(transformed, axis=1), maps


def center_membership(x: np.ndarray, centers: np.ndarray, temperature: float,
                      *, batch: int = 80_000) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Distance, hard id, soft weights and nearest-two squared-distance margin."""
    n, k = len(x), len(centers)
    hard = np.empty(n, dtype=np.int16)
    nearest = np.empty(n, dtype=np.float32)
    margin = np.empty(n, dtype=np.float32)
    soft = np.empty((n, k), dtype=np.float32)
    if CUDA_ENABLED:
        c = torch.as_tensor(centers, device="cuda", dtype=torch.float32)
        c2 = (c * c).sum(axis=1)
    else:
        c = centers
        c2 = np.sum(centers ** 2, axis=1)
    for start in range(0, n, batch):
        end = min(start + batch, n)
        if CUDA_ENABLED:
            xx = torch.as_tensor(x[start:end], device="cuda", dtype=torch.float32)
            d2 = torch.clamp((xx * xx).sum(axis=1, keepdim=True) + c2 - 2 * xx @ c.T, min=0)
            two, idx = torch.topk(d2, 2, dim=1, largest=False)
            weight = torch.softmax(-d2 / temperature, dim=1)
            hard[start:end] = idx[:, 0].cpu().numpy()
            nearest[start:end] = torch.sqrt(two[:, 0]).cpu().numpy()
            margin[start:end] = (two[:, 1] - two[:, 0]).cpu().numpy()
            soft[start:end] = weight.cpu().numpy()
        else:
            xx = x[start:end]
            d2 = np.maximum(np.sum(xx ** 2, axis=1, keepdims=True) + c2 - 2 * xx @ c.T, 0)
            order = np.argpartition(d2, 2, axis=1)[:, :2]
            two = np.take_along_axis(d2, order, axis=1)
            order = np.take_along_axis(order, np.argsort(two, axis=1), axis=1)
            two.sort(axis=1)
            logits = -d2 / temperature
            logits -= logits.max(axis=1, keepdims=True)
            w = np.exp(logits)
            hard[start:end] = order[:, 0]
            nearest[start:end] = np.sqrt(two[:, 0])
            margin[start:end] = two[:, 1] - two[:, 0]
            soft[start:end] = w / w.sum(axis=1, keepdims=True)
    return hard, nearest, soft, margin


def fit_region_probabilities(y: np.ndarray, symbol: np.ndarray, weight: np.ndarray,
                             *, alpha: float = 700, symbol_alpha: float = 1000) -> tuple[list[np.ndarray], dict[int, list[np.ndarray]]]:
    """Taskwise Dirichlet shrinkage for soft or one-hot region memberships."""
    k = weight.shape[1]
    global_p = []
    local: dict[int, list[np.ndarray]] = {}
    for task, classes in enumerate(TASK_SIZES):
        onehot = np.eye(classes, dtype=np.float32)[y[:, task]]
        background = (onehot.sum(axis=0) + 1) / (len(y) + classes)
        count = weight.T @ onehot
        total = weight.sum(axis=0)
        global_p.append(((count + alpha * background[None, :]) / (total[:, None] + alpha)).astype(np.float32))
    for s in np.unique(symbol):
        mask = symbol == s
        local[int(s)] = []
        w = weight[mask]
        for task, classes in enumerate(TASK_SIZES):
            onehot = np.eye(classes, dtype=np.float32)[y[mask, task]]
            count = w.T @ onehot
            total = w.sum(axis=0)
            local[int(s)].append(((count + symbol_alpha * global_p[task]) /
                                  (total[:, None] + symbol_alpha)).astype(np.float32))
    return global_p, local


def region_predict(weight: np.ndarray, symbol: np.ndarray,
                   general: list[np.ndarray], local: dict[int, list[np.ndarray]]) -> list[np.ndarray]:
    out = [np.empty((len(weight), c), dtype=np.float32) for c in TASK_SIZES]
    for s in np.unique(symbol):
        use = symbol == s
        source = local.get(int(s), general)
        for task in range(3):
            out[task][use] = weight[use] @ source[task]
    return out


def compact_features(x2: np.ndarray, symbols: np.ndarray, past_rv: np.ndarray,
                     n_symbols: int = 12) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    """Only pre-observation summaries; shape geometry is the six 16-wide groups."""
    if x2.shape[1] != 100:
        raise ValueError("X2 representation must have 100 coordinates")
    groups = ("vol_strength", "vol_asymmetry", "path_geometry", "trend", "efficiency", "participation")
    parts, names = [], []
    for j, group in enumerate(groups):
        block = x2[:, 16 * j:16 * (j + 1)]
        parts += [block[:, :4].mean(axis=1), (block[:, 2:4] - block[:, 4:6]).mean(axis=1),
                  (block[:, 2:4] - block[:, 10:12]).mean(axis=1)]
        names += [f"{group}_current", f"{group}_change4h", f"{group}_change1h"]
    own = np.stack(parts, axis=1).astype(np.float32)
    # All 12 assets share the same completed 15m grid, verified by the caller.
    if len(x2) % n_symbols:
        raise ValueError("unbalanced symbol panel")
    length = len(x2) // n_symbols
    if not np.array_equal(symbols.reshape(n_symbols, length)[:, 0], np.arange(n_symbols)):
        raise ValueError("expected symbol-major aligned panel")
    market = own.reshape(n_symbols, length, len(names))[:, :, ::3].mean(axis=0)
    market = np.tile(market, (n_symbols, 1))
    clock = x2[:, -4:]
    logrv = np.log(np.maximum(past_rv, 1e-10)).astype(np.float32)[:, None]
    own_names = names.copy()
    names += [f"market_{g}" for g in groups]
    names += ["clock_hour_sin", "clock_hour_cos", "clock_week_sin", "clock_week_cos", "log_past_rv"]
    features = np.column_stack((own, market, clock, logrv)).astype(np.float32)
    return features, names, {"own": np.arange(len(own_names)),
                             "market": np.arange(len(own_names), len(own_names) + 6),
                             "clock": np.arange(len(own_names) + 6, len(own_names) + 10),
                             "rv": np.asarray([len(own_names) + 10])}


def infer_compact_geometry(panel, cfg: dict, reference_maps: dict,
                           conditional_bundle: dict, geometry_bundle: dict,
                           rows: np.ndarray | None = None) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
    """Feature-only v2 inference on aligned, symbol-major completed snapshots.

    The panel includes at least 16 past 15m snapshots per symbol so ordered
    paths can be formed. The frozen per-symbol empirical maps are supplied;
    this function never reads outcome labels or refits probabilities.
    """
    if rows is None:
        rows = np.arange(len(panel))
    rep = make_representation(panel, cfg, reference_maps=reference_maps)
    symbols = panel.symbol_code.to_numpy(dtype=np.int8)
    compact, _, _ = compact_features(rep["X2_ordered_path"], symbols,
                                     panel.rv_slow_15m.to_numpy())
    _, distance, weight, margin = center_membership(rep["X1_endpoints"][rows, :-4],
                                                     geometry_bundle["centers"],
                                                     geometry_bundle["temperature"])
    asset = np.eye(int(symbols.max()) + 1, dtype=np.float32)[symbols[rows]]
    x = np.column_stack((compact[rows], asset, distance, margin)).astype(np.float32)
    models = conditional_bundle["models"]
    probabilities = [model.predict_proba(x).astype(np.float32) for model in models]
    return probabilities, {"geometry_distance": distance, "geometry_margin": margin,
                           "soft_membership": weight}
