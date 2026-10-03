"""Interpretable, finite market-pattern prototypes and mixed-semantic similarity.

This is a research representation layer. Prototype directions and semantic
centers are declared before examining their forward outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Coordinate:
    center: float  # percentile in historical training reference, 0..1
    half_width: float  # flat low-loss region around the economic state
    weight: float


@dataclass(frozen=True)
class Prototype:
    name: str
    direction: int
    mechanism: str
    coordinates: dict[str, Coordinate]


PROTOTYPES = (
    Prototype("trend_continuation_long", +1, "上行路径效率与主动参与配合，趋势仍在扩展", {
        "trend_15": Coordinate(.84, .12, 2.0), "efficiency_15": Coordinate(.80, .15, 1.4),
        "flow": Coordinate(.70, .20, 1.0), "path": Coordinate(.77, .19, 1.0),
        "vol_intensity": Coordinate(.65, .24, .7), "trend_change": Coordinate(.72, .23, .7),
    }),
    Prototype("trend_continuation_short", -1, "下行路径效率与主动卖出配合，趋势仍在扩展", {
        "trend_15": Coordinate(.16, .12, 2.0), "efficiency_15": Coordinate(.80, .15, 1.4),
        "flow": Coordinate(.30, .20, 1.0), "path": Coordinate(.23, .19, 1.0),
        "vol_intensity": Coordinate(.65, .24, .7), "trend_change": Coordinate(.28, .23, .7),
    }),
    Prototype("compression_release_long", +1, "波动从收缩转向扩张，路径和主动流向上确认", {
        "vol_intensity": Coordinate(.57, .22, 1.0), "vol_change": Coordinate(.87, .14, 1.8),
        "trend_change": Coordinate(.83, .16, 1.6), "path": Coordinate(.78, .19, 1.0),
        "flow": Coordinate(.67, .20, .9), "efficiency_15": Coordinate(.62, .23, .8),
    }),
    Prototype("compression_release_short", -1, "波动从收缩转向扩张，路径和主动流向下确认", {
        "vol_intensity": Coordinate(.57, .22, 1.0), "vol_change": Coordinate(.87, .14, 1.8),
        "trend_change": Coordinate(.17, .16, 1.6), "path": Coordinate(.22, .19, 1.0),
        "flow": Coordinate(.33, .20, .9), "efficiency_15": Coordinate(.62, .23, .8),
    }),
    Prototype("shock_rejection_long", +1, "下行冲击后长下影并高位收回，研究被吸收后的回升", {
        "trend_15": Coordinate(.23, .18, 1.2), "vol_intensity": Coordinate(.85, .15, 1.2),
        "lower_wick": Coordinate(.86, .13, 1.6), "path": Coordinate(.75, .20, 1.2),
        "flow": Coordinate(.60, .25, .7),
    }),
    Prototype("shock_rejection_short", -1, "上行冲击后长上影并低位收回，研究被吸收后的回落", {
        "trend_15": Coordinate(.77, .18, 1.2), "vol_intensity": Coordinate(.85, .15, 1.2),
        "upper_wick": Coordinate(.86, .13, 1.6), "path": Coordinate(.25, .20, 1.2),
        "flow": Coordinate(.40, .25, .7),
    }),
)


class QuantileReference:
    """Training-only empirical mapping into comparable percentile coordinates."""

    def __init__(self, frame: pd.DataFrame, columns: list[str]):
        self.knots = {}
        probs = np.linspace(0, 1, 1001)
        for name in columns:
            values = frame[name].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                raise ValueError(f"no training values for {name}")
            knots = np.quantile(values, probs)
            unique, first = np.unique(knots, return_index=True)
            self.knots[name] = (unique, probs[first])

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        output = pd.DataFrame(index=frame.index)
        for name, (knots, probs) in self.knots.items():
            output[name] = np.interp(frame[name].to_numpy(dtype=float), knots, probs, left=0.0, right=1.0)
            output.loc[~np.isfinite(frame[name].to_numpy(dtype=float)), name] = np.nan
        return output


def similarity(percentiles: pd.DataFrame, prototype: Prototype, *, distance_scale: float = .20) -> np.ndarray:
    """Flat-core, prototype-specific weighted distance; missing core means no match."""
    if distance_scale <= 0:
        raise ValueError("distance_scale must be positive")
    dist = np.zeros(len(percentiles), dtype=float)
    missing = np.zeros(len(percentiles), dtype=bool)
    total = 0.0
    for name, spec in prototype.coordinates.items():
        value = percentiles[name].to_numpy(dtype=float)
        missing |= ~np.isfinite(value)
        deviation = np.abs(value - spec.center)
        outside = np.maximum(deviation - spec.half_width, 0.0)
        # A small curvature inside the acceptable region preserves the broad
        # plateau but prevents a point mass at similarity == 1.
        inner = 0.02 * (deviation / max(spec.half_width, 1e-6)) ** 2
        dist += spec.weight * ((outside / distance_scale) ** 2 + inner)
        total += spec.weight
    output = np.exp(-dist / total)
    output[missing] = np.nan
    return output


def sparse_events(frame: pd.DataFrame, mask: np.ndarray, *, min_gap_hours: int = 4) -> pd.DataFrame:
    """Keep first match after each symbol's refractory gap, in time order."""
    selected = frame.loc[mask].sort_values(["symbol", "available_at"]).copy()
    keep = np.zeros(len(selected), dtype=bool)
    for _, indices in selected.groupby("symbol", sort=False).indices.items():
        last = None
        for idx in indices:
            current = selected.available_at.iloc[idx]
            if last is None or current - last >= pd.Timedelta(hours=min_gap_hours):
                keep[idx] = True
                last = current
    return selected.iloc[np.flatnonzero(keep)].copy()
