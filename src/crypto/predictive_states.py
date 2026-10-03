"""Direction-neutral geometric states with time-separated distribution fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import kmeans_plusplus

from .full_patterns import EmpiricalMap


TASK_SIZES = (5, 5, 2)
PATH_LAGS = (4, 8, 12, 16)
CUDA_ENABLED = torch.cuda.is_available()


def assignment(x: np.ndarray, centers: np.ndarray, *, batch: int = 100_000) -> tuple[np.ndarray, np.ndarray]:
    """Nearest actual geometry center; return squared distance without a huge N x K matrix."""
    ids = np.empty(len(x), dtype=np.int16)
    distance = np.empty(len(x), dtype=np.float32)
    if CUDA_ENABLED:
        gpu_centers = torch.as_tensor(centers, device="cuda", dtype=torch.float32)
        gpu_center_norm = (gpu_centers * gpu_centers).sum(dim=1)
    else:
        center_norm = np.sum(centers.astype(np.float64) ** 2, axis=1)
    for start in range(0, len(x), batch):
        if CUDA_ENABLED:
            part = torch.as_tensor(x[start:start + batch], device="cuda", dtype=torch.float32)
            d2 = torch.clamp((part * part).sum(dim=1, keepdim=True) + gpu_center_norm - 2 * part @ gpu_centers.T, min=0)
            value, label = torch.min(d2, dim=1)
            ids[start:start + batch] = label.cpu().numpy()
            distance[start:start + batch] = torch.sqrt(value).cpu().numpy()
        else:
            part = x[start:start + batch].astype(np.float64, copy=False)
            d2 = np.maximum(np.sum(part ** 2, axis=1, keepdims=True) + center_norm - 2 * part @ centers.T, 0)
            ids[start:start + batch] = np.argmin(d2, axis=1)
            distance[start:start + batch] = np.sqrt(np.min(d2, axis=1))
    return ids, distance


def fit_centers(x: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    """Small-k centroid search: CUDA distance and CUDA center updates when available."""
    if not CUDA_ENABLED:
        return MiniBatchKMeans(n_clusters=n_clusters, batch_size=4096, n_init=2,
                               max_iter=120, random_state=seed).fit(x).cluster_centers_.astype(np.float32)
    rng = np.random.default_rng(seed)
    init_sample = x[rng.choice(len(x), size=min(30_000, len(x)), replace=False)]
    initial, _ = kmeans_plusplus(init_sample, n_clusters=n_clusters, random_state=seed)
    data = torch.as_tensor(x, device="cuda", dtype=torch.float32)
    centers = torch.as_tensor(initial, device="cuda", dtype=torch.float32)
    for _ in range(35):
        distance = torch.clamp((data * data).sum(dim=1, keepdim=True)
                               + (centers * centers).sum(dim=1) - 2 * data @ centers.T, min=0)
        label = torch.argmin(distance, dim=1)
        sums = torch.zeros_like(centers)
        sums.index_add_(0, label, data)
        counts = torch.bincount(label, minlength=n_clusters)
        updated = torch.where(counts[:, None] > 0, sums / counts.clamp_min(1)[:, None], centers)
        shift = torch.max(torch.abs(updated - centers)).item()
        centers = updated
        if shift < 1e-5:
            break
    return centers.cpu().numpy().astype(np.float32)


def outcome_labels(return4: np.ndarray, return24: np.ndarray, rv4: np.ndarray,
                   past_rv24: np.ndarray, edges: tuple[float, ...]) -> np.ndarray:
    scale4 = np.sqrt(np.maximum(past_rv24 / 6, 1e-10))
    scale24 = np.sqrt(np.maximum(past_rv24, 1e-10))
    return np.stack((np.digitize(return4 / scale4, edges),
                     np.digitize(return24 / scale24, edges),
                     (rv4 > past_rv24 / 6).astype(np.int8)), axis=1).astype(np.int8)


def distribution(labels: np.ndarray, prior: list[np.ndarray] | None = None,
                 strength: float = 0) -> list[np.ndarray]:
    answer = []
    for task, classes in enumerate(TASK_SIZES):
        counts = np.bincount(labels[:, task], minlength=classes).astype(np.float64)
        if prior is None:
            probability = (counts + 1) / (counts.sum() + classes)
        else:
            probability = (counts + strength * prior[task]) / (counts.sum() + strength)
        answer.append(probability)
    return answer


def log_loss(labels: np.ndarray, probabilities: list[np.ndarray]) -> np.ndarray:
    return np.stack([-np.log(np.clip(p[np.arange(len(labels)), labels[:, j]], 1e-8, 1))
                     for j, p in enumerate(probabilities)], axis=1)


def scalar_loss(labels: np.ndarray, probabilities: list[np.ndarray]) -> float:
    return float(log_loss(labels, probabilities).mean())


def fit_reference_maps(panel: pd.DataFrame, cfg: dict) -> dict[str, EmpiricalMap]:
    """Persistable pre-2025 marginal transforms for later feature-only inference."""
    fields = tuple(dict.fromkeys(name for names in cfg["representation_columns"].values() for name in names))
    cutoff = pd.Timestamp(cfg["discovery_fit_end"])
    rng = np.random.default_rng(cfg["random_seed"])
    references = {}
    for symbol, index in panel.groupby("symbol", sort=False).indices.items():
        index = np.asarray(index)
        train_index = index[(panel.available_at.iloc[index] < cutoff).to_numpy()]
        sample = rng.choice(train_index, size=min(len(train_index), 25_000), replace=False)
        references[str(symbol)] = EmpiricalMap(panel.iloc[sample], fields, sample=25_000)
    return references


def make_representation(panel: pd.DataFrame, cfg: dict, *, pooled: bool = False,
                        reference_maps: dict[str, EmpiricalMap] | None = None) -> dict[str, np.ndarray]:
    """Fit 101-knot midrank maps on pre-2025 features only, then build ordered paths."""
    if pooled and reference_maps is not None:
        raise ValueError("external per-symbol references cannot be pooled")
    groups = cfg["representation_columns"]
    fields = tuple(dict.fromkeys(name for names in groups.values() for name in names))
    n = len(panel)
    mapped = np.empty((n, len(fields)), dtype=np.float32)
    missing = np.empty_like(mapped, dtype=bool)
    cutoff = pd.Timestamp(cfg["discovery_fit_end"])
    rng = np.random.default_rng(cfg["random_seed"])
    if pooled:
        fit_indices = np.flatnonzero((panel.available_at < cutoff).to_numpy())
        fit_indices = rng.choice(fit_indices, size=min(len(fit_indices), 90_000), replace=False)
        reference = EmpiricalMap(panel.iloc[fit_indices], fields, sample=90_000)
    for symbol, index in panel.groupby("symbol", sort=False).indices.items():
        index = np.asarray(index)
        if reference_maps is not None:
            reference = reference_maps[str(symbol)]
        elif not pooled:
            train_index = index[(panel.available_at.iloc[index] < cutoff).to_numpy()]
            sample = rng.choice(train_index, size=min(len(train_index), 25_000), replace=False)
            reference = EmpiricalMap(panel.iloc[sample], fields, sample=25_000)
        mapped[index], missing[index] = reference.transform(panel.iloc[index])

    field_pos = {name: j for j, name in enumerate(fields)}
    x0_blocks, x1_blocks, x2_blocks = [], [], []
    for group, names in groups.items():
        current = mapped[:, [field_pos[name] for name in names]]
        x0_blocks.append((group, current))
        x1_parts = [current]
        x2_parts = [current]
        if group in cfg["path_columns"]:
            path_pos = [field_pos[name] for name in cfg["path_columns"][group]]
            history = {}
            for lag in PATH_LAGS:
                past = np.full((n, len(path_pos)), .5, dtype=np.float32)
                for _, index in panel.groupby("symbol", sort=False).indices.items():
                    index = np.asarray(index)
                    past[index[lag:]] = mapped[index[:-lag]][:, path_pos]
                history[lag] = past
            x1_parts.append(history[16])
            x2_parts.extend(history[lag] for lag in (16, 12, 8, 4))
            nodes = np.stack((history[16], history[12], history[8], history[4], mapped[:, path_pos]), axis=1)
            x2_parts.extend((nodes.argmax(axis=1).astype(np.float32) / 4,
                             nodes.argmin(axis=1).astype(np.float32) / 4))
        x1_blocks.append((group, np.concatenate(x1_parts, axis=1)))
        x2_blocks.append((group, np.concatenate(x2_parts, axis=1)))

    def balanced(blocks: list[tuple[str, np.ndarray]]) -> np.ndarray:
        # Each economic group has unit total squared scale, irrespective of path width.
        return np.concatenate([((values - .5) * 2 / np.sqrt(values.shape[1])).astype(np.float32)
                               for _, values in blocks], axis=1)

    return {"X0_current": balanced(x0_blocks), "X1_endpoints": balanced(x1_blocks),
            "X2_ordered_path": balanced(x2_blocks), "missing_count": missing.sum(axis=1)}


@dataclass
class StateModel:
    root_centers: np.ndarray
    child_centers: dict[int, np.ndarray]
    root_probs: dict[int, list[np.ndarray]]
    leaf_probs: dict[int, list[np.ndarray]]
    root_symbol_probs: dict[tuple[int, int], list[np.ndarray]]
    leaf_symbol_probs: dict[tuple[int, int], list[np.ndarray]]
    background: dict[int, list[np.ndarray]]
    root_radius: dict[int, float]
    leaf_radius: dict[int, float]
    medoids: list[dict]
    split_evidence: list[dict]


def _fit_probabilities(labels: np.ndarray, symbols: np.ndarray, cluster: np.ndarray,
                       indices: np.ndarray, parent: dict[int, list[np.ndarray]], background: dict[int, list[np.ndarray]],
                       *, strength: float, symbol_strength: float) -> tuple[dict[int, list[np.ndarray]], dict[tuple[int, int], list[np.ndarray]]]:
    general = {}
    by_symbol = {}
    for state in np.unique(cluster[indices]):
        subset = indices[cluster[indices] == state]
        general[int(state)] = distribution(labels[subset], parent[int(state)], strength)
        for symbol in np.unique(symbols[subset]):
            local = subset[symbols[subset] == symbol]
            by_symbol[(int(state), int(symbol))] = distribution(labels[local], general[int(state)], symbol_strength)
    return general, by_symbol


def _predict_table(ids: np.ndarray, symbols: np.ndarray, general: dict[int, list[np.ndarray]],
                   local: dict[tuple[int, int], list[np.ndarray]], background: dict[int, list[np.ndarray]],
                   accepted: np.ndarray) -> list[np.ndarray]:
    out = [np.empty((len(ids), classes), dtype=np.float32) for classes in TASK_SIZES]
    for state in np.unique(ids):
        positions = np.flatnonzero(ids == state)
        for symbol in np.unique(symbols[positions]):
            use = positions[symbols[positions] == symbol]
            p = local.get((int(state), int(symbol)), general.get(int(state), background[int(symbol)]))
            for j in range(3):
                out[j][use] = p[j]
    for symbol in np.unique(symbols[~accepted]):
        use = np.flatnonzero((symbols == symbol) & ~accepted)
        for j in range(3):
            out[j][use] = background[int(symbol)][j]
    return out


def fit_states(x: np.ndarray, labels: np.ndarray, symbols: np.ndarray, available_at: pd.Series,
               fit_idx: np.ndarray, inner_idx: np.ndarray, discovery_idx: np.ndarray,
               cfg: dict) -> StateModel:
    rng = np.random.default_rng(cfg["random_seed"])
    geometry = []
    for symbol in np.unique(symbols[fit_idx]):
        positions = fit_idx[symbols[fit_idx] == symbol]
        geometry.extend(rng.choice(positions, size=min(len(positions), cfg["geometry_sample_per_symbol"]), replace=False))
    geometry = np.asarray(geometry, dtype=np.int64)
    centers = fit_centers(x[geometry], cfg["root_clusters"], cfg["random_seed"])
    fit_root, _ = assignment(x[fit_idx], centers)
    inner_root, _ = assignment(x[inner_idx], centers)
    all_root, all_distance = assignment(x[discovery_idx], centers)
    bg_fit = distribution(labels[fit_idx])
    prior_fit = {k: bg_fit for k in range(cfg["root_clusters"])}
    fit_map = np.full(len(x), -1, dtype=np.int16)
    fit_map[fit_idx] = fit_root
    inner_map = np.full(len(x), -1, dtype=np.int16)
    inner_map[inner_idx] = inner_root
    root_fit, _ = _fit_probabilities(labels, symbols, fit_map, fit_idx, prior_fit, {},
                                    strength=cfg["shrinkage"], symbol_strength=cfg["symbol_shrinkage"])
    children: dict[int, np.ndarray] = {}
    split_evidence = []
    for root in range(cfg["root_clusters"]):
        train = fit_idx[fit_root == root]
        check = inner_idx[inner_root == root]
        candidate_geometry = geometry[assignment(x[geometry], centers)[0] == root]
        if len(train) < 2 * cfg["minimum_leaf_training"] or len(check) < 2 * cfg["minimum_leaf_selection"] or len(candidate_geometry) < 1000:
            split_evidence.append({"root": root, "accepted": False, "reason": "insufficient_support"})
            continue
        child_centers = fit_centers(x[candidate_geometry], 2, cfg["random_seed"] + root + 1)
        train_child, _ = assignment(x[train], child_centers)
        check_child, _ = assignment(x[check], child_centers)
        if min(np.bincount(train_child, minlength=2)) < cfg["minimum_leaf_training"] or min(np.bincount(check_child, minlength=2)) < cfg["minimum_leaf_selection"]:
            split_evidence.append({"root": root, "accepted": False, "reason": "small_child"})
            continue
        child_prob = {c: distribution(labels[train[train_child == c]], root_fit[root], cfg["shrinkage"])
                      for c in (0, 1)}
        parent_pred = [np.repeat(root_fit[root][j][None, :], len(check), axis=0) for j in range(3)]
        child_pred = [np.stack([child_prob[int(c)][j] for c in check_child]) for j in range(3)]
        improvement = scalar_loss(labels[check], parent_pred) - scalar_loss(labels[check], child_pred)
        days = [int(available_at.iloc[check[check_child == c]].dt.floor("D").nunique()) for c in (0, 1)]
        accepted = improvement > .0005 and min(days) >= 20
        split_evidence.append({"root": root, "accepted": bool(accepted), "inner_logloss_gain": improvement,
                               "train_children": np.bincount(train_child, minlength=2).tolist(),
                               "inner_children": np.bincount(check_child, minlength=2).tolist(), "inner_days": days})
        if accepted:
            children[root] = child_centers

    root_all = np.full(len(x), -1, dtype=np.int16)
    root_all[discovery_idx] = all_root
    leaf_all = root_all.copy()
    leaf_distance = np.zeros(len(discovery_idx), dtype=np.float32)
    for root in range(cfg["root_clusters"]):
        local = np.flatnonzero(all_root == root)
        if root in children:
            c, d = assignment(x[discovery_idx[local]], children[root])
            leaf_all[discovery_idx[local]] = root * 2 + c + 100
            leaf_distance[local] = d
        else:
            leaf_all[discovery_idx[local]] = root
            leaf_distance[local] = all_distance[local]

    background = {int(s): distribution(labels[discovery_idx[symbols[discovery_idx] == s]],
                                        distribution(labels[discovery_idx]), cfg["symbol_shrinkage"])
                  for s in np.unique(symbols[discovery_idx])}
    global_bg = distribution(labels[discovery_idx])
    root_parent = {r: global_bg for r in range(cfg["root_clusters"])}
    root_probs, root_symbol_probs = _fit_probabilities(labels, symbols, root_all, discovery_idx,
                                                        root_parent, background, strength=cfg["shrinkage"],
                                                        symbol_strength=cfg["symbol_shrinkage"])
    leaf_parent = {int(leaf): root_probs[int(leaf) if leaf < 100 else (int(leaf) - 100) // 2]
                   for leaf in np.unique(leaf_all[discovery_idx])}
    leaf_probs, leaf_symbol_probs = _fit_probabilities(labels, symbols, leaf_all, discovery_idx,
                                                        leaf_parent, background, strength=cfg["shrinkage"],
                                                        symbol_strength=cfg["symbol_shrinkage"])
    root_radius = {int(r): float(np.quantile(all_distance[all_root == r], cfg["reject_radius_quantile"]))
                   for r in np.unique(all_root)}
    leaf_radius = {int(leaf): float(np.quantile(leaf_distance[leaf_all[discovery_idx] == leaf], cfg["reject_radius_quantile"]))
                   for leaf in np.unique(leaf_all[discovery_idx])}
    medoids = []
    geom_root, _ = assignment(x[geometry], centers)
    for leaf in np.unique(leaf_all[discovery_idx]):
        leaf = int(leaf)
        if leaf >= 100:
            root = (leaf - 100) // 2
            center = children[root][(leaf - 100) % 2]
        else:
            root = leaf
            center = centers[root]
        candidates = geometry[geom_root == root]
        if leaf >= 100:
            c_id, _ = assignment(x[candidates], children[root])
            candidates = candidates[c_id == (leaf - 100) % 2]
        near = candidates[np.argmin(np.sum((x[candidates] - center) ** 2, axis=1))]
        rows = discovery_idx[leaf_all[discovery_idx] == leaf]
        medoids.append({"leaf": leaf, "parent": root, "medoid_row": int(near),
                        "medoid_symbol": int(symbols[near]), "medoid_available_at": str(available_at.iloc[near]),
                        "support_rows": len(rows), "support_days": int(available_at.iloc[rows].dt.floor("D").nunique()),
                        "radius90": leaf_radius[leaf]})
    return StateModel(centers, children, root_probs, leaf_probs, root_symbol_probs,
                      leaf_symbol_probs, background, root_radius, leaf_radius, medoids, split_evidence)


def predict_states(model: StateModel, x: np.ndarray, symbols: np.ndarray,
                   *, split: bool) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    root, root_distance = assignment(x, model.root_centers)
    ids = root.copy()
    distance = root_distance.copy()
    if split:
        for parent, centers in model.child_centers.items():
            rows = np.flatnonzero(root == parent)
            child, d = assignment(x[rows], centers)
            ids[rows] = parent * 2 + child + 100
            distance[rows] = d
    radius = model.leaf_radius if split else model.root_radius
    accepted = np.fromiter((distance[j] <= radius[int(state)] for j, state in enumerate(ids)),
                           dtype=bool, count=len(ids))
    probs = _predict_table(ids, symbols, model.leaf_probs if split else model.root_probs,
                           model.leaf_symbol_probs if split else model.root_symbol_probs,
                           model.background, accepted)
    return probs, ids, distance, accepted
