"""Shared causal features for detector branches."""

from __future__ import annotations

import numpy as np
import pandas as pd


def grouped_shift_rolling(series: pd.Series, roots: pd.Series, window: int, op: str) -> pd.Series:
    grouped = series.groupby(roots, sort=False)
    return grouped.transform(lambda x: getattr(x.shift(1).rolling(window, min_periods=window), op)())


def breakout_features(frame: pd.DataFrame, lookback: int, confirm: int, alpha: float) -> pd.DataFrame:
    roots = frame["root"]
    ref_high = grouped_shift_rolling(frame["high"], roots, lookback, "max")
    ref_low = grouped_shift_rolling(frame["low"], roots, lookback, "min")
    recent_high = frame["high"].groupby(roots, sort=False).transform(
        lambda x: x.rolling(confirm, min_periods=confirm).max()
    )
    recent_low = frame["low"].groupby(roots, sort=False).transform(
        lambda x: x.rolling(confirm, min_periods=confirm).min()
    )
    baseline_volume = grouped_shift_rolling(frame["volume"], roots, lookback, "mean")
    recent_volume = frame["volume"].groupby(roots, sort=False).transform(
        lambda x: x.rolling(confirm, min_periods=confirm).mean()
    )
    volume_ratio = recent_volume / baseline_volume.replace(0, np.nan)
    up_breach = (recent_high - ref_high) / frame["close"]
    down_breach = (ref_low - recent_low) / frame["close"]
    threshold = alpha * frame["sigma_hat"]
    side = np.select([up_breach.ge(threshold), down_breach.ge(threshold)], [1, -1], default=0)
    strength = np.maximum(up_breach, down_breach).div(threshold.replace(0, np.nan))
    return pd.DataFrame({
        "ref_high": ref_high, "ref_low": ref_low, "volume_ratio": volume_ratio,
        "breakout_side": side, "breakout_strength": strength,
    }, index=frame.index)


def rolling_percentile(series: pd.Series, roots: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    minimum = min_periods or window
    return series.groupby(roots, sort=False).transform(
        lambda x: x.rolling(window, min_periods=minimum).apply(
            lambda values: (values[:-1] <= values[-1]).mean() if len(values) > 1 else np.nan,
            raw=True,
        )
    )
