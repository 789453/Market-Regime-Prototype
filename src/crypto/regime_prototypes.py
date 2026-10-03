"""Outcome-tested geometric prototypes and state-dependent pair decisions.

All arrays passed here are already aligned to completed, hourly snapshots.  The
functions do not access prices or future labels except in explicit fit calls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.tree import DecisionTreeRegressor

from .signal_chain import beta_neutral_pair


REGIMES = ("calm_down", "calm_up", "normal", "high")
ENTER_QUANTILES = {"calm_down": .52, "calm_up": .58, "normal": .70, "high": .80}
EXIT_QUANTILES = {"calm_down": .25, "calm_up": .30, "normal": .35, "high": .40}


def future_path_labels(log_open: np.ndarray, entry: np.ndarray, horizons=(4, 24, 72, 120)) -> dict[str, np.ndarray]:
    """Labels from the next executable open; side moments exclude its prior return."""
    delta = np.diff(log_open, prepend=log_open[0])
    up_prefix = np.r_[0.0, np.cumsum(np.maximum(delta, 0.0) ** 2)]
    down_prefix = np.r_[0.0, np.cumsum(np.minimum(delta, 0.0) ** 2)]
    out = {}
    for hours in horizons:
        bars = hours * 12
        end = entry + bars
        good = (entry >= 0) & (end < len(log_open))
        ret = np.full(len(entry), np.nan)
        up = np.full(len(entry), np.nan)
        down = np.full(len(entry), np.nan)
        ret[good] = log_open[end[good]] - log_open[entry[good]]
        up[good] = up_prefix[end[good] + 1] - up_prefix[entry[good] + 1]
        down[good] = down_prefix[end[good] + 1] - down_prefix[entry[good] + 1]
        out[f"return{hours}"] = ret
        out[f"up{hours}"] = np.maximum(up, 0)
        out[f"down{hours}"] = np.maximum(down, 0)
    return out


def semivariance_direction(up: np.ndarray, down: np.ndarray) -> np.ndarray:
    """Bounded direction of future realized movement, separate from total risk."""
    return np.divide(up - down, up + down + 1e-12)


def select_beta_pair(prediction: np.ndarray, beta: np.ndarray) -> tuple[int, int, float]:
    """Maximize forecast of the weights that would actually be held."""
    b = np.asarray(beta, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    pair = (b[None, :] * p[:, None] - b[:, None] * p[None, :]) / (b[:, None] + b[None, :])
    np.fill_diagonal(pair, -np.inf)
    i, j = np.unravel_index(np.argmax(pair), pair.shape)
    return int(i), int(j), float(pair[i, j])


def held_pair_score(prediction: np.ndarray, beta: np.ndarray, long: int, short: int) -> float:
    return float(beta[short] * prediction[long] - beta[long] * prediction[short]) / float(beta[long] + beta[short])


def classify_regime(market_rv: np.ndarray, market_trend: np.ndarray,
                    low: float, high: float) -> np.ndarray:
    code = np.full(len(market_rv), 2, dtype=np.int8)
    calm = market_rv < low
    code[calm & (market_trend < 0)] = 0
    code[calm & (market_trend >= 0)] = 1
    code[market_rv >= high] = 3
    return code


@dataclass
class PrototypeFit:
    geometry: MiniBatchKMeans
    parent_means: np.ndarray
    leaves: list[DecisionTreeRegressor | None]
    leaf_means: list[dict[int, np.ndarray]]
    accepted: np.ndarray
    split_features: np.ndarray
    train_counts: np.ndarray

    def predict(self, x: np.ndarray, geometry_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        group = self.geometry.predict(geometry_x).astype(np.int16)
        result = self.parent_means[group].copy()
        leaf = np.full(len(x), -1, dtype=np.int16)
        for k, tree in enumerate(self.leaves):
            if tree is None:
                continue
            where = np.flatnonzero(group == k)
            if not len(where):
                continue
            node = tree.apply(x[where])
            leaf[where] = node.astype(np.int16)
            result[where] = np.stack([self.leaf_means[k][int(v)] for v in node])
        return result, group, leaf


def _shrunk_means(y: np.ndarray, group: np.ndarray, global_mean: np.ndarray,
                  n_groups: int, strength: float = 500.0) -> tuple[np.ndarray, np.ndarray]:
    count = np.bincount(group, minlength=n_groups).astype(np.int32)
    sums = np.zeros((n_groups, y.shape[1]), dtype=np.float64)
    np.add.at(sums, group, y)
    means = (sums + strength * global_mean) / (count[:, None] + strength)
    return means.astype(np.float32), count


def fit_predictive_prototypes(x: np.ndarray, geometry_x: np.ndarray,
                              y: np.ndarray, dates: np.ndarray,
                              *, clusters: int = 12, seed: int = 20260930,
                              fit_task_weights: np.ndarray | None = None,
                              certify_tasks: tuple[int, ...] | None = None) -> PrototypeFit:
    """Geometric parents; keep an outcome split only if it beats parent later in time.

    The multi-target signature is supplied by the caller.  Tree leaves are
    geometric regions of X and membership never depends on a current label.
    """
    if len(x) != len(geometry_x) or len(x) != len(y):
        raise ValueError("misaligned prototype arrays")
    if not np.isfinite(x).all() or not np.isfinite(geometry_x).all() or not np.isfinite(y).all():
        raise ValueError("nonfinite prototype input")
    weights = (np.ones(y.shape[1]) if fit_task_weights is None else
               np.asarray(fit_task_weights, dtype=np.float64))
    if weights.shape != (y.shape[1],) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("invalid task weights")
    tasks = tuple(range(y.shape[1])) if certify_tasks is None else certify_tasks
    if not tasks or min(tasks) < 0 or max(tasks) >= y.shape[1]:
        raise ValueError("invalid certification tasks")
    rng = np.random.default_rng(seed)
    subsample = rng.choice(len(x), size=min(100_000, len(x)), replace=False)
    km = MiniBatchKMeans(n_clusters=clusters, batch_size=4096, n_init=3,
                         max_iter=120, random_state=seed).fit(geometry_x[subsample])
    group = km.predict(geometry_x).astype(np.int16)
    global_mean = y.mean(axis=0)
    parent, count = _shrunk_means(y, group, global_mean, clusters)
    leaves: list[DecisionTreeRegressor | None] = [None] * clusters
    leaf_means: list[dict[int, np.ndarray]] = [{} for _ in range(clusters)]
    accepted = np.zeros(clusters, dtype=bool)
    features = np.full(clusters, -1, dtype=np.int16)
    # Time separation is shared by all coins: no random row split.
    unique_dates = np.unique(dates)
    cut = unique_dates[max(1, int(len(unique_dates) * .7) - 1)]
    for k in range(clusters):
        indices = np.flatnonzero(group == k)
        early = indices[dates[indices] <= cut]
        later = indices[dates[indices] > cut]
        if len(early) < 800 or len(later) < 300:
            continue
        minimum = max(160, int(len(early) * .12))
        candidate = DecisionTreeRegressor(max_depth=1, min_samples_leaf=minimum,
                                          random_state=seed + k).fit(x[early], y[early] * weights)
        if candidate.tree_.node_count < 3:
            continue
        early_leaf = candidate.apply(x[early])
        later_leaf = candidate.apply(x[later])
        if len(np.unique(early_leaf)) != 2 or len(np.unique(later_leaf)) != 2:
            continue
        # At least twenty separate UTC days per child in both responsibilities.
        if any(len(np.unique(dates[early[early_leaf == node]])) < 20 or
               len(np.unique(dates[later[later_leaf == node]])) < 20
               for node in np.unique(early_leaf)):
            continue
        local_parent = (y[early].sum(axis=0) + 200 * global_mean) / (len(early) + 200)
        estimates = {}
        for node in np.unique(early_leaf):
            sample = y[early[early_leaf == node]]
            estimates[int(node)] = (sample.sum(axis=0) + 250 * local_parent) / (len(sample) + 250)
        before = np.mean((y[later][:, tasks] - local_parent[list(tasks)]) ** 2)
        after = np.mean((y[later][:, tasks] -
                         np.stack([estimates[int(v)] for v in later_leaf])[:, tasks]) ** 2)
        if not after < before:
            continue
        # After selection, refit only on the already eligible training history.
        final = DecisionTreeRegressor(max_depth=1,
                                      min_samples_leaf=max(160, int(len(indices) * .12)),
                                      random_state=seed + k).fit(x[indices], y[indices] * weights)
        if final.tree_.node_count < 3:
            continue
        node_all = final.apply(x[indices])
        fitted_means = {}
        for node in np.unique(node_all):
            sample = y[indices[node_all == node]]
            fitted_means[int(node)] = ((sample.sum(axis=0) + 250 * parent[k]) /
                                       (len(sample) + 250)).astype(np.float32)
        leaves[k], leaf_means[k] = final, fitted_means
        accepted[k] = True
        features[k] = int(final.tree_.feature[0])
    return PrototypeFit(km, parent, leaves, leaf_means, accepted, features, count)


def adaptive_pair_weights(prediction: np.ndarray, beta: np.ndarray,
                          regime: np.ndarray, enter: np.ndarray, leave: np.ndarray,
                          *, continuous: bool = True, min_hours: int = 3,
                          max_hours: int = 120, review_every_hours: int = 1,
                          switch_hurdle: float = .0004,
                          rebalance_deadband: float = 0.) -> tuple[np.ndarray, np.ndarray]:
    """State-dependent thresholds with the held pair's own survival score."""
    n, coins = prediction.shape
    weights = np.zeros((n, coins), dtype=np.float64)
    chosen = np.full((n, 3), -1, dtype=np.float64)
    old_long = old_short = -1
    age = 0
    if review_every_hours < 1 or switch_hurdle < 0 or rebalance_deadband < 0:
        raise ValueError("invalid policy review controls")
    for t in range(n):
        if t % review_every_hours and old_long >= 0 and age < max_hours:
            age += 1
            weights[t] = weights[t - 1]
            chosen[t] = (old_long, old_short, age)
            continue
        if t % review_every_hours and old_long < 0:
            continue
        cand_long, cand_short, best = select_beta_pair(prediction[t], beta[t])
        e = max(float(enter[t]), 0.0)
        l = max(float(leave[t]), 0.0)
        own = (held_pair_score(prediction[t], beta[t], old_long, old_short)
               if old_long >= 0 else -np.inf)
        if old_long >= 0:
            age += 1
            if own <= 0 or age >= max_hours or (age >= min_hours and own < l):
                old_long = old_short = -1
                age = 0
        if old_long < 0 and best >= e and best > 0:
            old_long, old_short = cand_long, cand_short
            age = 1
            own = best
        elif old_long >= 0 and age >= min_hours and best > max(e, own + switch_hurdle):
            old_long, old_short = cand_long, cand_short
            age = 1
            own = best
        if old_long >= 0:
            # The hurdle controls entry; size responds smoothly to a positive
            # forecast, while remaining bounded under very low risk.
            scale = min(1.0, max(.0, own) / max(e * 1.5, .0003)) if continuous else 1.0
            if scale >= .05:
                target = scale * beta_neutral_pair(old_long, old_short, beta[t])
                if (t > 0 and chosen[t - 1, 0] == old_long and chosen[t - 1, 1] == old_short
                    and np.abs(target - weights[t - 1]).sum() < rebalance_deadband):
                    target = weights[t - 1]
                weights[t] = target
                chosen[t] = (old_long, old_short, age)
    return weights, chosen
