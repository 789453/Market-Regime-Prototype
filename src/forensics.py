"""Stage 1 forensic statistics computed from immutable raw bars."""

from __future__ import annotations

from math import gcd
from functools import reduce

import numpy as np
import pandas as pd


PRICE_COLUMNS = ("open", "high", "low", "close")


def roll_boundary_table(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(["root", "timestamp"], kind="stable").reset_index(drop=True)
    grouped = ordered.groupby("root", sort=False, observed=True)
    previous_symbol = grouped["local_symbol"].shift()
    previous_time = grouped["timestamp"].shift()
    previous_close = grouped["close"].shift()
    boundary = previous_symbol.notna() & ordered["local_symbol"].ne(previous_symbol)
    result = pd.DataFrame(
        {
            "root": ordered.loc[boundary, "root"].to_numpy(),
            "previous_symbol": previous_symbol.loc[boundary].to_numpy(),
            "next_symbol": ordered.loc[boundary, "local_symbol"].to_numpy(),
            "previous_timestamp": previous_time.loc[boundary].to_numpy(),
            "next_timestamp": ordered.loc[boundary, "timestamp"].to_numpy(),
            "previous_close": previous_close.loc[boundary].to_numpy(),
            "next_open": ordered.loc[boundary, "open"].to_numpy(),
        }
    )
    result["gap_minutes"] = (
        (result["next_timestamp"] - result["previous_timestamp"]).dt.total_seconds() / 60
    ).astype(int)
    result["jump_bp"] = np.log(result["next_open"] / result["previous_close"]) * 10_000
    return result.reset_index(drop=True)


def _grid_gcd(values: np.ndarray, scale: int = 100) -> float:
    integers = np.unique(np.rint(values * scale).astype(np.int64))
    differences = np.diff(integers)
    differences = differences[differences > 0]
    return reduce(gcd, differences.tolist()) / scale


def infer_tick_sizes(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for root, group in frame.groupby("root", sort=True, observed=True):
        values = group.loc[:, PRICE_COLUMNS].to_numpy().ravel()
        tick = _grid_gcd(values)
        median_close = float(group["close"].median())
        rows.append(
            {
                "root": root,
                "tick_size": tick,
                "median_close": median_close,
                "tick_bp": tick / median_close * 10_000,
            }
        )
    return pd.DataFrame(rows)


def ohlc_violations(frame: pd.DataFrame) -> pd.Series:
    lower = frame[["open", "close"]].min(axis=1)
    upper = frame[["open", "close"]].max(axis=1)
    return ~(
        frame["low"].le(lower)
        & lower.le(upper)
        & upper.le(frame["high"])
        & frame["wap"].between(frame["low"], frame["high"])
    )


def anomaly_table(frame: pd.DataFrame, window: int = 390, threshold: float = 8.0) -> pd.DataFrame:
    rows = []
    for root, group in frame.groupby("root", sort=True, observed=True):
        group = group.sort_values("timestamp").copy()
        previous_close = group["close"].shift()
        consecutive = group["timestamp"].diff().eq(pd.Timedelta(minutes=1))
        same_contract = group["local_symbol"].eq(group["local_symbol"].shift())
        valid_pair = (~group["is_filled"]) & (~group["is_filled"].shift(fill_value=True))
        returns = np.log(group["close"] / previous_close).where(consecutive & same_contract & valid_pair)
        median = returns.rolling(window, min_periods=60).median().shift(1)
        mad = (returns - median).abs().rolling(window, min_periods=60).median().shift(1)
        robust_scale = 1.4826 * mad
        score = (returns - median).abs() / robust_scale.replace(0, np.nan)
        next_return = returns.shift(periods=-1)
        flagged = score.gt(threshold)
        if flagged.any():
            part = group.loc[flagged, ["timestamp", "local_symbol", "open", "high", "low", "close"]].copy()
            part.insert(0, "root", root)
            part["ret_bp"] = returns.loc[flagged].to_numpy() * 10_000
            part["next_ret_bp"] = next_return.loc[flagged].to_numpy() * 10_000
            part["round_trip_ratio"] = (
                (returns.loc[flagged] + next_return.loc[flagged]).abs()
                / returns.loc[flagged].abs()
            ).to_numpy()
            part["robust_score"] = score.loc[flagged].to_numpy()
            rows.append(part)
    if not rows:
        return pd.DataFrame(columns=["root", "timestamp", "local_symbol", *PRICE_COLUMNS, "ret_bp", "robust_score"])
    return pd.concat(rows, ignore_index=True).sort_values("robust_score", ascending=False)


def gap_summary(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame.loc[~frame["is_filled"]].sort_values(["root", "timestamp"]).copy()
    valid["gap_minutes"] = valid.groupby("root", observed=True)["timestamp"].diff().dt.total_seconds().div(60)
    valid["category"] = pd.cut(
        valid["gap_minutes"],
        bins=[0, 1, 30, 24 * 60, np.inf],
        labels=["1m", "2-30m", "30m-1d", ">1d"],
        include_lowest=True,
    )
    return (
        valid.dropna(subset=["category"])
        .groupby(["root", "category"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )


def missing_weekdays(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for root, group in frame.groupby("root", sort=True, observed=True):
        observed = pd.DatetimeIndex(pd.to_datetime(pd.unique(group["session_id"])))
        expected = pd.bdate_range(observed.min(), observed.max())
        for missing in expected.difference(observed):
            rows.append({"root": root, "session_id": missing.date()})
    return pd.DataFrame(rows)


def lead_lag_correlations(bars: pd.DataFrame, max_lag: int = 10) -> pd.DataFrame:
    overlap = bars.loc[
        bars["root"].isin(["ES", "HSI"])
        & bars["timestamp"].dt.hour.between(9, 18)
    ]
    wide = overlap.pivot(index="timestamp", columns="root", values="ret").dropna()
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        pair = pd.concat([wide["ES"].shift(lag), wide["HSI"]], axis=1).dropna()
        rows.append({"lag": lag, "correlation": pair.iloc[:, 0].corr(pair.iloc[:, 1]), "n": len(pair)})
    return pd.DataFrame(rows)


def effective_instrument_count(bars: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    overlap = bars.loc[bars["timestamp"].dt.hour.between(9, 18)]
    wide = overlap.pivot(index="timestamp", columns="root", values="ret").dropna()
    correlation = wide.corr()
    eigenvalues = np.linalg.eigvalsh(correlation.to_numpy())
    n_eff = float(eigenvalues.sum() ** 2 / np.square(eigenvalues).sum())
    t_eff = 0.5 * n_eff
    sharpe_se = float(np.sqrt(1 / t_eff))
    return correlation, n_eff, sharpe_se
