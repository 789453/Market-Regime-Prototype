"""Online volatility-scaled ZigZag and causal swing features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def online_zigzag(frame: pd.DataFrame, kappa: float = 2.0, n_ref: int = 12) -> pd.DataFrame:
    """Confirm pivots only after a volatility-scaled reversal is observable."""
    rows: list[dict] = []
    for root, group in frame.sort_values(["root", "timestamp"], kind="stable").groupby("root", sort=False):
        direction = 0
        extreme_price = np.nan
        extreme_time = None
        extreme_pos = 0
        last_pivot_price = np.nan
        last_pivot_pos = 0
        last_pivot_volume = np.nan
        for pos, row in enumerate(group.itertuples(index=False)):
            if not row.is_valid or not np.isfinite(row.close) or not np.isfinite(row.sigma_hat):
                continue
            threshold = kappa * row.sigma_hat * np.sqrt(n_ref)
            log_close = np.log(row.close)
            if not np.isfinite(extreme_price):
                extreme_price, extreme_time, extreme_pos = log_close, row.timestamp, pos
                last_pivot_price, last_pivot_pos = log_close, pos
                continue
            if direction >= 0:
                if log_close >= extreme_price:
                    extreme_price, extreme_time, extreme_pos = log_close, row.timestamp, pos
                if extreme_price - log_close > threshold:
                    duration = max(extreme_pos - last_pivot_pos, 1)
                    rows.append({
                        "root": root, "event_time": extreme_time, "confirm_time": row.timestamp,
                        "pivot_type": "peak", "pivot_price": np.exp(extreme_price),
                        "direction": 1, "duration_bars": duration,
                        "amp_sigma": abs(extreme_price - last_pivot_price) / max(row.sigma_hat * np.sqrt(duration), 1e-12),
                        "volume_at_confirm": row.volume,
                    })
                    last_pivot_price, last_pivot_pos, last_pivot_volume = extreme_price, extreme_pos, row.volume
                    direction = -1
                    extreme_price, extreme_time, extreme_pos = log_close, row.timestamp, pos
            else:
                if log_close <= extreme_price:
                    extreme_price, extreme_time, extreme_pos = log_close, row.timestamp, pos
                if log_close - extreme_price > threshold:
                    duration = max(extreme_pos - last_pivot_pos, 1)
                    rows.append({
                        "root": root, "event_time": extreme_time, "confirm_time": row.timestamp,
                        "pivot_type": "trough", "pivot_price": np.exp(extreme_price),
                        "direction": -1, "duration_bars": duration,
                        "amp_sigma": abs(extreme_price - last_pivot_price) / max(row.sigma_hat * np.sqrt(duration), 1e-12),
                        "volume_at_confirm": row.volume,
                    })
                    last_pivot_price, last_pivot_pos, last_pivot_volume = extreme_price, extreme_pos, row.volume
                    direction = 1
                    extreme_price, extreme_time, extreme_pos = log_close, row.timestamp, pos
    columns = [
        "root", "event_time", "confirm_time", "pivot_type", "pivot_price",
        "direction", "duration_bars", "amp_sigma", "volume_at_confirm",
    ]
    pivots = pd.DataFrame(rows, columns=columns)
    if not pivots.empty and (pivots["confirm_time"] < pivots["event_time"]).any():
        raise AssertionError("pivot confirmed before its event")
    return pivots


def causal_structure_features(frame: pd.DataFrame, kappa: float = 2.0) -> pd.DataFrame:
    """Map the last three confirmed pivots to each bar without backdating."""
    pivots = online_zigzag(frame, kappa=kappa)
    output = frame[["root", "timestamp"]].copy()
    for column in ("trend_direction", "retrace_ratio", "time_ratio", "volume_ratio", "bars_since_confirm"):
        output[column] = np.nan
    for root, group in frame.groupby("root", sort=False):
        root_pivots = pivots.loc[pivots["root"].eq(root)].sort_values("confirm_time").copy()
        if len(root_pivots) < 3:
            continue
        previous_price = root_pivots["pivot_price"].shift(1)
        prior_price = root_pivots["pivot_price"].shift(2)
        advance = np.log(previous_price / prior_price).abs()
        pullback = np.log(root_pivots["pivot_price"] / previous_price).abs()
        root_pivots["trend_direction"] = np.where(
            root_pivots["pivot_type"].eq("peak"),
            np.where(root_pivots["pivot_price"] > prior_price, 1, -1),
            np.where(root_pivots["pivot_price"] < prior_price, -1, 1),
        )
        root_pivots["retrace_ratio"] = pullback / advance.replace(0, np.nan)
        root_pivots["time_ratio"] = root_pivots["duration_bars"] / root_pivots["duration_bars"].shift(1)
        root_pivots["volume_ratio"] = root_pivots["volume_at_confirm"] / root_pivots["volume_at_confirm"].shift(1).replace(0, np.nan)
        left = group[["timestamp"]].copy()
        left["original_index"] = group.index
        right = root_pivots[["confirm_time", "trend_direction", "retrace_ratio", "time_ratio", "volume_ratio"]].dropna(subset=["retrace_ratio"])
        mapped = pd.merge_asof(
            left.sort_values("timestamp"), right.sort_values("confirm_time"),
            left_on="timestamp", right_on="confirm_time", direction="backward",
        )
        mapped["bars_since_confirm"] = (mapped["timestamp"] - mapped["confirm_time"]) / pd.Timedelta(minutes=5)
        mapped = mapped.set_index("original_index")
        columns = ["trend_direction", "retrace_ratio", "time_ratio", "volume_ratio", "bars_since_confirm"]
        output.loc[mapped.index, columns] = mapped[columns]
    return output
