"""Contract-local intraday and overnight returns."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_returns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.sort_values(["root", "timestamp"], kind="stable").reset_index(drop=True).copy()
    grouped = result.groupby("root", sort=False, observed=True)
    previous_close = grouped["close"].shift()
    previous_valid = grouped["is_valid"].shift(fill_value=False)
    previous_symbol = grouped["local_symbol"].shift()
    previous_session = grouped["session_id"].shift()

    result["is_roll_boundary"] = previous_symbol.notna() & result["local_symbol"].ne(previous_symbol)
    both_valid = result["is_valid"] & previous_valid
    same_contract = result["local_symbol"].eq(previous_symbol)
    same_session = result["session_id"].eq(previous_session)
    raw_return = np.log(result["close"] / previous_close)
    result["ret"] = raw_return.where(both_valid & same_contract & same_session)
    result["overnight_ret"] = raw_return.where(both_valid & same_contract & ~same_session)
    return result


def _future_window_sum(values: pd.Series, horizon: int) -> pd.Series:
    return values.iloc[::-1].rolling(horizon, min_periods=horizon).sum().iloc[::-1]


def build_labels(frame: pd.DataFrame, horizons: tuple[int, ...] = (6, 12, 24, 48, 96)) -> pd.DataFrame:
    ordered = frame.sort_values(["root", "timestamp"], kind="stable").reset_index(drop=True)
    output = ordered[["root", "timestamp"]].copy()
    for horizon in horizons:
        fwd_return = pd.Series(np.nan, index=ordered.index, dtype=float)
        valid_label = pd.Series(False, index=ordered.index, dtype=bool)
        for _, index in ordered.groupby("root", sort=False, observed=True).groups.items():
            group = ordered.loc[index]
            future_close = group["close"].shift(periods=-horizon)
            candidate = np.log(future_close / group["close"])
            future_valid_count = _future_window_sum(group["is_valid"].astype(int), horizon + 1)
            roll_count = _future_window_sum(group["is_roll_boundary"].shift(periods=-1, fill_value=False).astype(int), horizon)
            session_change = group["session_id"].ne(group["session_id"].shift()).astype(int)
            future_session_changes = _future_window_sum(session_change.shift(periods=-1, fill_value=0), horizon)
            valid = (
                future_close.notna()
                & future_valid_count.ge(0.6 * (horizon + 1))
                & roll_count.eq(0)
                & future_session_changes.le(1)
            )
            fwd_return.loc[index] = candidate.where(valid)
            valid_label.loc[index] = valid
        output[f"fwd_ret_{horizon}"] = fwd_return
        output[f"fwd_ret_z_{horizon}"] = fwd_return / (ordered["sigma_hat"] * np.sqrt(horizon))
        output[f"label_valid_{horizon}"] = valid_label
    return output
