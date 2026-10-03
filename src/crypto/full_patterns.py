"""Full-field, group-balanced market prototypes for rolling research.

All 112 completed 5m/15m feature values enter the geometry. A small semantic
seed locates candidate historical paths; the center and dispersion are then
learned from those paths without using their subsequent returns. Outcomes may
only adjust strongly shrunken group reliability inside a training fold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import GROUPS


BASE_COLUMNS = tuple(f"{name}_{freq}" for freq in ("5m", "15m") for group in GROUPS.values() for name in group)
PATH_REPRESENTATIVES = {
    "vol_strength": ("rv_fast_slow_15m", "rv_surprise_5m"),
    "vol_asymmetry": ("semivar_balance_15m", "lower_wick_med_5m"),
    "path_geometry": ("close_location_5m", "vwap_distance_15m"),
    "trend": ("momentum_z_med_15m", "direction_alignment_5m"),
    "efficiency": ("efficiency_med_15m", "sign_flip_med_5m"),
    "participation": ("flow_imb_med_15m", "quote_rel_fast_5m"),
}
PATH_COLUMNS = tuple(f"{name}__{suffix}" for names in PATH_REPRESENTATIVES.values() for name in names for suffix in ("lag4h", "delta4h"))
MODEL_COLUMNS = BASE_COLUMNS + PATH_COLUMNS
assert len(BASE_COLUMNS) == 112 and len(MODEL_COLUMNS) == 136

COLUMN_GROUP = {}
for group, names in GROUPS.items():
    for freq in ("5m", "15m"):
        for name in names:
            COLUMN_GROUP[f"{name}_{freq}"] = group
for group, names in PATH_REPRESENTATIVES.items():
    for name in names:
        for suffix in ("lag4h", "delta4h"):
            COLUMN_GROUP[f"{name}__{suffix}"] = group
GROUP_COLUMNS = {group: tuple(name for name in MODEL_COLUMNS if COLUMN_GROUP[name] == group) for group in GROUPS}


@dataclass(frozen=True)
class Seed:
    name: str
    family: str
    direction: int
    coordinates: dict[str, float]
    prior_group_weights: dict[str, float]


def _weights(**kwargs: float) -> dict[str, float]:
    return {g: kwargs.get(g, .45 if g != "time_context" else .15) for g in GROUPS}


SEEDS = (
    Seed("trend_long", "trend_continuation", 1, {
        "momentum_z_med_15m": .84, "momentum_z_med_15m__lag4h": .70,
        "efficiency_med_15m": .78, "direction_alignment_5m": .83,
        "flow_imb_med_15m": .68, "rv_fast_slow_15m": .59,
    }, _weights(trend=2.0, efficiency=1.45, participation=1.0, path_geometry=.8, vol_strength=.6)),
    Seed("trend_short", "trend_continuation", -1, {
        "momentum_z_med_15m": .16, "momentum_z_med_15m__lag4h": .30,
        "efficiency_med_15m": .78, "direction_alignment_5m": .17,
        "flow_imb_med_15m": .32, "rv_fast_slow_15m": .59,
    }, _weights(trend=2.0, efficiency=1.45, participation=1.0, path_geometry=.8, vol_strength=.6)),
    Seed("release_long", "compression_release", 1, {
        "rv_fast_slow_15m__lag4h": .25, "rv_fast_slow_15m": .68,
        "rv_fast_slow_15m__delta4h": .84, "momentum_z_med_15m": .76,
        "close_location_5m": .78, "flow_imb_med_15m": .68,
    }, _weights(vol_strength=2.0, trend=1.25, path_geometry=1.1, participation=.9, efficiency=.7)),
    Seed("release_short", "compression_release", -1, {
        "rv_fast_slow_15m__lag4h": .25, "rv_fast_slow_15m": .68,
        "rv_fast_slow_15m__delta4h": .84, "momentum_z_med_15m": .24,
        "close_location_5m": .22, "flow_imb_med_15m": .32,
    }, _weights(vol_strength=2.0, trend=1.25, path_geometry=1.1, participation=.9, efficiency=.7)),
    Seed("rejection_long", "shock_rejection", 1, {
        "momentum_z_med_15m__lag4h": .20, "lower_wick_med_5m": .86,
        "close_location_5m": .78, "rv_surprise_5m": .82,
        "flow_imb_med_15m": .55,
    }, _weights(vol_strength=1.1, vol_asymmetry=1.6, path_geometry=1.65, trend=.8, participation=.7)),
    Seed("rejection_short", "shock_rejection", -1, {
        "momentum_z_med_15m__lag4h": .80, "upper_wick_med_5m": .86,
        "close_location_5m": .22, "rv_surprise_5m": .82,
        "flow_imb_med_15m": .45,
    }, _weights(vol_strength=1.1, vol_asymmetry=1.6, path_geometry=1.65, trend=.8, participation=.7)),
    Seed("range_long", "range_reversion", 1, {
        "efficiency_med_15m": .22, "sign_flip_med_15m": .79,
        "vwap_distance_15m": .18, "close_location_5m": .62,
        "return_acf1_med_15m": .28,
    }, _weights(efficiency=1.9, path_geometry=1.75, vol_strength=.6, trend=.55, participation=.55)),
    Seed("range_short", "range_reversion", -1, {
        "efficiency_med_15m": .22, "sign_flip_med_15m": .79,
        "vwap_distance_15m": .82, "close_location_5m": .38,
        "return_acf1_med_15m": .28,
    }, _weights(efficiency=1.9, path_geometry=1.75, vol_strength=.6, trend=.55, participation=.55)),
)


def add_path(frame: pd.DataFrame) -> pd.DataFrame:
    """Create only historical 4h levels and changes, per symbol."""
    out = frame.sort_values(["symbol", "available_at"]).copy()
    for names in PATH_REPRESENTATIVES.values():
        for name in names:
            previous = out.groupby("symbol", sort=False)[name].shift(16)
            out[f"{name}__lag4h"] = previous.astype("float32")
            out[f"{name}__delta4h"] = (out[name] - previous).astype("float32")
    return out


class EmpiricalMap:
    """Training-only marginal map; compact quantile knots bound memory."""

    def __init__(self, train: pd.DataFrame, columns: tuple[str, ...], *, sample: int = 60000):
        self.columns = columns
        source = train.iloc[::max(1, len(train) // sample)]
        probs = np.linspace(0, 1, 101)
        self.knots: list[tuple[np.ndarray, np.ndarray]] = []
        for name in columns:
            values = source[name].to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            if not len(values):
                raise ValueError(f"empty training feature {name}")
            knots = np.quantile(values, probs)
            unique, first, counts = np.unique(knots, return_index=True, return_counts=True)
            if len(unique) == 1:
                self.knots.append((unique, np.array([.5])))
            else:
                # A zero-inflated or binary feature has tied quantile knots.
                # Map the tie to its mid-rank, not to the first CDF point;
                # otherwise common zeros appear spuriously extreme.
                midrank = (probs[first] + probs[first + counts - 1]) / 2
                self.knots.append((unique, midrank))

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        out = np.empty((len(frame), len(self.columns)), dtype=np.float32)
        missing = np.empty_like(out, dtype=np.bool_)
        for j, name in enumerate(self.columns):
            value = frame[name].to_numpy(dtype=np.float64)
            finite = np.isfinite(value)
            knots, probs = self.knots[j]
            if len(knots) == 1:
                out[:, j] = .5
            else:
                out[:, j] = np.interp(value, knots, probs, left=0, right=1)
            out[~finite, j] = .5
            missing[:, j] = ~finite
        return out, missing


def anchor_score(mapped: np.ndarray, seed: Seed) -> np.ndarray:
    index = {name: j for j, name in enumerate(MODEL_COLUMNS)}
    losses = [(mapped[:, index[name]] - target) ** 2 for name, target in seed.coordinates.items()]
    return -np.mean(losses, axis=0)


def anchor_indices(mapped: np.ndarray, seed: Seed, *, fraction: float = .008) -> np.ndarray:
    score = anchor_score(mapped, seed)
    n = max(200, int(len(score) * fraction))
    n = min(n, len(score))
    return np.argpartition(score, -n)[-n:]


@dataclass
class FullPrototype:
    seed: Seed
    center: np.ndarray
    width: np.ndarray
    contrast: np.ndarray
    redundancy: np.ndarray
    group_prior: np.ndarray
    anchor_count: int


def fit_prototype(mapped: np.ndarray, seed: Seed, *, redundancy_sample: np.ndarray) -> FullPrototype:
    chosen = anchor_indices(mapped, seed)
    examples = mapped[chosen]
    center = np.median(examples, axis=0).astype(np.float32)
    q25, q75 = np.quantile(examples, [.25, .75], axis=0)
    width = np.maximum(.07, (q75 - q25) * .55).astype(np.float32)
    contrast = (.25 + 2.5 * np.abs(center - .5)).astype(np.float32)
    redundancy = np.ones(len(MODEL_COLUMNS), dtype=np.float32)
    for names in GROUP_COLUMNS.values():
        pos = np.array([MODEL_COLUMNS.index(n) for n in names])
        x = redundancy_sample[:, pos]
        corr = np.corrcoef(x, rowvar=False)
        corr = np.nan_to_num(corr, nan=0)
        np.fill_diagonal(corr, 0)
        redundancy[pos] = 1 / (1 + np.maximum(np.abs(corr) - .7, 0).sum(axis=0))
    prior = np.array([seed.prior_group_weights[g] for g in GROUPS], dtype=np.float32)
    return FullPrototype(seed, center, width, contrast, redundancy, prior, len(chosen))


def group_distances(mapped: np.ndarray, missing: np.ndarray, proto: FullPrototype, *, use_redundancy: bool) -> np.ndarray:
    """Every field enters one of seven group distances; missing gets a penalty."""
    out = np.empty((len(mapped), len(GROUPS)), dtype=np.float32)
    for g, names in enumerate(GROUP_COLUMNS.values()):
        pos = np.array([MODEL_COLUMNS.index(n) for n in names])
        deviation = np.abs(mapped[:, pos] - proto.center[pos])
        exterior = np.maximum(deviation - proto.width[pos], 0)
        loss = .015 * (deviation / proto.width[pos]) ** 2 + (exterior / .25) ** 2
        loss += missing[:, pos] * .25
        weight = proto.contrast[pos] * (proto.redundancy[pos] if use_redundancy else 1)
        out[:, g] = (loss * weight).sum(axis=1) / weight.sum()
    return out


def combine_distances(distances: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.exp(-(distances @ weights) / weights.sum()).astype(np.float32)


def dynamic_weights(distances: np.ndarray, signed_outcome: np.ndarray, dates: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Small, training-only reliability adjustment that requires both halves."""
    midpoint = np.median(dates.astype("datetime64[ns]").astype("int64"))
    early = dates.astype("datetime64[ns]").astype("int64") < midpoint
    result = prior.astype(np.float64).copy()
    for g in range(distances.shape[1]):
        lifts = []
        for mask in (early, ~early):
            valid = mask & np.isfinite(signed_outcome)
            if valid.sum() < 100:
                lifts.append(0.0)
                continue
            threshold = np.quantile(distances[valid, g], .08)
            near = valid & (distances[:, g] <= threshold)
            lifts.append(float(np.mean(signed_outcome[near]) - np.mean(signed_outcome[valid])))
        persistent = max(0.0, min(lifts))
        result[g] *= .75 + .5 * np.tanh(persistent / .001)
    return result.astype(np.float32)


def cosine_scores(mapped: np.ndarray, center: np.ndarray) -> np.ndarray:
    x = mapped * 2 - 1
    c = center * 2 - 1
    norm = np.linalg.norm(x, axis=1) * max(float(np.linalg.norm(c)), 1e-8)
    return np.divide(x @ c, norm, out=np.zeros(len(x), dtype=np.float32), where=norm > 0)
