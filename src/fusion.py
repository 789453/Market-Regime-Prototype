"""Causal confidence-weighted detector fusion for Stage 5."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def effective_detector_count(correlation: np.ndarray) -> float:
    """Participation-ratio effective dimension of a correlation matrix."""
    eig = np.clip(np.linalg.eigvalsh(np.asarray(correlation, dtype=float)), 0.0, None)
    denom = float(np.square(eig).sum())
    return float(eig.sum() ** 2 / denom) if denom > 0 else 0.0


def detector_dependence(
    signals: Mapping[str, pd.DataFrame], min_joint_triggers: int = 25
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Direction correlation on joint triggers and common-trigger Jaccard rate."""
    names = list(signals)
    directions = pd.concat(
        {name: frame["direction"].astype(float) for name, frame in signals.items()}, axis=1
    ).fillna(0.0)
    corr = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    common = corr.copy()
    for i, left in enumerate(names):
        x = directions[left].to_numpy()
        for j in range(i + 1, len(names)):
            right = names[j]
            y = directions[right].to_numpy()
            both = (x != 0) & (y != 0)
            either = (x != 0) | (y != 0)
            rho = 0.0
            if np.array_equal(x, y) and either.any():
                rho = 1.0
            elif both.sum() >= min_joint_triggers and np.std(x[both]) > 0 and np.std(y[both]) > 0:
                rho = float(np.corrcoef(x[both], y[both])[0, 1])
                if not np.isfinite(rho):
                    rho = 0.0
            rate = float(both.sum() / either.sum()) if either.any() else 0.0
            corr.loc[left, right] = corr.loc[right, left] = rho
            common.loc[left, right] = common.loc[right, left] = rate
    return corr, common


def cluster_detectors(correlation: pd.DataFrame, threshold: float = 0.60) -> pd.Series:
    """Complete-linkage clusters; absolute rho above threshold is redundant."""
    if len(correlation) == 1:
        return pd.Series(1, index=correlation.index, name="cluster", dtype=int)
    similarity = np.clip(np.abs(correlation.to_numpy(dtype=float)), 0.0, 1.0)
    np.fill_diagonal(similarity, 1.0)
    distances = squareform(1.0 - similarity, checks=False)
    tree = linkage(distances, method="complete")
    labels = fcluster(tree, t=1.0 - threshold, criterion="distance")
    return pd.Series(labels, index=correlation.index, name="cluster", dtype=int)


def _causal_active_votes(frame: pd.DataFrame, persist_horizon: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand each trigger only forward, with linear decay over its fixed horizon."""
    n = len(frame)
    numerator = np.zeros(n, dtype=float)
    denominator = np.zeros(n, dtype=float)
    horizon_mass = np.zeros(n, dtype=float)
    direction = frame["direction"].fillna(0.0).to_numpy(dtype=float)
    magnitude = frame["magnitude"].fillna(0.0).to_numpy(dtype=float)
    confidence = frame["confidence"].fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    gate = frame.get("gate_score", pd.Series(1.0, index=frame.index)).fillna(1.0).to_numpy(dtype=float)
    horizons = frame["horizon"].fillna(1).clip(lower=1).to_numpy(dtype=int)
    trigger_positions = np.flatnonzero((direction != 0) & (confidence > 0) & (magnitude > 0))
    for start in trigger_positions:
        length = int(horizons[start]) if persist_horizon else 1
        stop = min(n, start + length)
        decay = 1.0 - np.arange(stop - start, dtype=float) / length
        p = confidence[start]
        numerator[start:stop] += p * direction[start] * magnitude[start] * gate[start] * decay
        denominator[start:stop] += p * decay
        horizon_mass[start:stop] += p * horizons[start] * decay
    return numerator, denominator, horizon_mass


def fuse_signals(
    signals: Mapping[str, pd.DataFrame],
    tau: float = 0.5,
    correlation_threshold: float = 0.60,
    persist_horizon: bool = True,
    causal_clusters: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fuse votes after averaging redundant detectors within fixed clusters."""
    if not signals:
        raise ValueError("signals must not be empty")
    names = list(signals)
    reference = signals[names[0]].index
    if any(not reference.equals(frame.index) for frame in signals.values()):
        raise ValueError("all signal frames must have identical ordered indices")
    corr, common = detector_dependence(signals)
    clusters = cluster_detectors(corr, correlation_threshold)
    n = len(reference)
    num_by_detector = np.zeros((n, len(names)), dtype=float)
    den_by_detector = np.zeros_like(num_by_detector)
    hor_by_detector = np.zeros_like(num_by_detector)
    roots = reference.get_level_values("root").to_numpy()
    for j, name in enumerate(names):
        frame = signals[name]
        for root in pd.unique(roots):
            loc = np.flatnonzero(roots == root)
            num, den, hor = _causal_active_votes(frame.iloc[loc], persist_horizon)
            num_by_detector[loc, j] = num
            den_by_detector[loc, j] = den
            hor_by_detector[loc, j] = hor
    total_num = np.zeros(n, dtype=float)
    total_den = np.zeros(n, dtype=float)
    total_hor = np.zeros(n, dtype=float)
    active_clusters = np.zeros(n, dtype=np.int16)

    def apply_clusters(loc: np.ndarray, labels: np.ndarray) -> None:
        for cluster_id in np.unique(labels):
            members = np.flatnonzero(labels == cluster_id)
            total_num[loc] += num_by_detector[np.ix_(loc, members)].mean(axis=1)
            total_den[loc] += den_by_detector[np.ix_(loc, members)].mean(axis=1)
            total_hor[loc] += hor_by_detector[np.ix_(loc, members)].mean(axis=1)
            active_clusters[loc] += (
                den_by_detector[np.ix_(loc, members)].sum(axis=1) > 0
            ).astype(np.int16)

    if causal_clusters and len(names) > 1:
        # Freeze clusters for a UTC day using only joint triggers from earlier days.
        # This is causal and avoids unstable intraday re-clustering on one new event.
        count = np.zeros((len(names), len(names)), dtype=int)
        sum_x = np.zeros_like(count, dtype=float)
        sum_y = np.zeros_like(count, dtype=float)
        sum_xx = np.zeros_like(count, dtype=float)
        sum_yy = np.zeros_like(count, dtype=float)
        sum_xy = np.zeros_like(count, dtype=float)
        raw_direction = np.column_stack(
            [signals[name]["direction"].fillna(0.0).to_numpy(dtype=float) for name in names]
        )
        timestamps = reference.get_level_values("timestamp")
        day_values = timestamps.normalize().asi8
        order = np.argsort(day_values, kind="stable")
        sorted_days = day_values[order]
        boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_days)) + 1, n]
        labels = np.arange(1, len(names) + 1, dtype=int)
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            loc = order[left:right]
            apply_clusters(loc, labels)
            changed = False
            day_direction = raw_direction[loc]
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    both = (day_direction[:, i] != 0) & (day_direction[:, j] != 0)
                    if not both.any():
                        continue
                    x, y = day_direction[both, i], day_direction[both, j]
                    count[i, j] += len(x)
                    sum_x[i, j] += x.sum()
                    sum_y[i, j] += y.sum()
                    sum_xx[i, j] += np.square(x).sum()
                    sum_yy[i, j] += np.square(y).sum()
                    sum_xy[i, j] += (x * y).sum()
                    changed = True
            if changed:
                online = np.eye(len(names))
                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        c = count[i, j]
                        if c < 25:
                            continue
                        vx = sum_xx[i, j] - sum_x[i, j] ** 2 / c
                        vy = sum_yy[i, j] - sum_y[i, j] ** 2 / c
                        if vx > 0 and vy > 0:
                            rho = (sum_xy[i, j] - sum_x[i, j] * sum_y[i, j] / c) / np.sqrt(vx * vy)
                            online[i, j] = online[j, i] = np.clip(rho, -1.0, 1.0)
                labels = cluster_detectors(
                    pd.DataFrame(online, index=names, columns=names), correlation_threshold
                ).to_numpy()
    else:
        labels = clusters.to_numpy()
        apply_clusters(np.arange(n), labels)
    out = pd.DataFrame(index=reference)
    out["mu"] = total_num / (total_den + tau)
    out["confidence_mass"] = total_den
    out["active_horizon"] = np.divide(total_hor, total_den, out=np.ones(n), where=total_den > 0)
    out["active_detectors"] = (den_by_detector > 0).sum(axis=1).astype(np.int16)
    out["active_clusters"] = active_clusters
    dependency = pd.DataFrame(index=names)
    dependency["cluster"] = clusters
    dependency["mean_abs_direction_correlation"] = (np.abs(corr).sum(axis=1) - 1) / max(len(names) - 1, 1)
    dependency["mean_common_trigger_rate"] = (common.sum(axis=1) - 1) / max(len(names) - 1, 1)
    dependency.attrs["correlation"] = corr
    dependency.attrs["common_trigger_rate"] = common
    dependency.attrs["effective_detector_count"] = effective_detector_count(corr.to_numpy())
    return out, dependency
