"""Abdi-Ranaldo (CHL) effective spread and execution cost curves."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


DEFAULT_COMMISSION_BP = {"ES": 0.05, "NQ": 0.03, "RTY": 0.08, "HSI": 0.40, "HTI": 0.50}


def estimate_chl_spread(bars: pd.DataFrame, ticks: pd.DataFrame) -> pd.DataFrame:
    work = bars.sort_values(["root", "timestamp"], kind="stable").copy()
    work["local_hour"] = (work["local_minute"] // 60).astype(int)
    close_log = np.log(work["close"])
    eta = (np.log(work["high"]) + np.log(work["low"])) / 2
    grouped = work.groupby(["root", "session_id"], sort=False, observed=True)
    eta_next = eta.groupby([work["root"], work["session_id"]], sort=False).shift(periods=-1)
    next_valid = grouped["is_valid"].shift(periods=-1, fill_value=False)
    pair_valid = work["is_valid"] & next_valid
    work["chl_product"] = ((close_log - eta) * (close_log - eta_next)).where(pair_valid)

    curve = (
        work.groupby(["root", "local_hour"], observed=True)["chl_product"]
        .agg(chl_mean="mean", valid_pairs="count")
        .reset_index()
    )
    curve["chl_raw_bp"] = 2 * np.sqrt(curve["chl_mean"].clip(lower=0)) * 10_000
    curve = curve.merge(ticks[["root", "tick_bp"]], on="root", how="left", validate="many_to_one")
    curve["spread_bp"] = curve[["chl_raw_bp", "tick_bp"]].max(axis=1)
    return curve


def build_cost_model(
    spread_curve: pd.DataFrame,
    *,
    cost_multiplier: float = 1.5,
    slip_ticks: float = 0.5,
) -> dict:
    instruments = {}
    for root, group in spread_curve.groupby("root", sort=True, observed=True):
        tick_bp = float(group["tick_bp"].iloc[0])
        commission = DEFAULT_COMMISSION_BP[root]
        hourly = []
        for row in group.itertuples(index=False):
            raw_round_trip = row.spread_bp + 2 * slip_ticks * tick_bp + 2 * commission
            hourly.append(
                {
                    "local_hour": int(row.local_hour),
                    "spread_bp": round(float(row.spread_bp), 6),
                    "valid_pairs": int(row.valid_pairs),
                    "round_trip_cost_bp": round(float(raw_round_trip * cost_multiplier), 6),
                }
            )
        instruments[root] = {
            "tick_bp": round(tick_bp, 6),
            "commission_bp_per_side": commission,
            "slippage_ticks_per_side": slip_ticks,
            "hourly": hourly,
        }
    return {"cost_multiplier": cost_multiplier, "estimator": "Abdi-Ranaldo CHL with one-tick floor", "instruments": instruments}


def write_cost_model(model: dict, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")


def causal_chl_spread(bars: pd.DataFrame, ticks: pd.DataFrame, min_pairs: int = 20) -> pd.Series:
    """Estimate each bar's spread from CHL pairs observable by that close."""
    output = pd.Series(np.nan, index=bars.index, dtype=float, name="spread_bp")
    tick_map = ticks.set_index("root")["tick_bp"].to_dict()
    computed: dict[object, float] = {}
    for root, group in bars.groupby("root", sort=False, observed=True):
        group = group.sort_values("timestamp")
        sums = np.zeros(24, dtype=float)
        counts = np.zeros(24, dtype=int)
        previous = None
        for row in group.itertuples():
            if (
                previous is not None
                and previous.is_valid
                and row.is_valid
                and previous.session_id == row.session_id
            ):
                c_previous = np.log(previous.close)
                eta_previous = (np.log(previous.high) + np.log(previous.low)) / 2
                eta_current = (np.log(row.high) + np.log(row.low)) / 2
                product = (c_previous - eta_previous) * (c_previous - eta_current)
                previous_hour = int(previous.local_minute // 60)
                sums[previous_hour] += product
                counts[previous_hour] += 1
            hour = int(row.local_minute // 60)
            raw = 2 * np.sqrt(max(sums[hour] / counts[hour], 0)) * 10_000 if counts[hour] >= min_pairs else 0.0
            computed[row.Index] = max(raw, tick_map[root])
            previous = row
    output.loc[list(computed)] = list(computed.values())
    return output
