"""Valid-bar masking and causal 1-minute to 5-minute aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_valid_mask(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    real = result["bar_count"].gt(0) & result["volume"].gt(0)
    flat = (
        result["open"].eq(result["high"])
        & result["high"].eq(result["low"])
        & result["low"].eq(result["close"])
    )
    result["is_real_1m"] = real
    result["is_flat"] = flat
    result["is_filled"] = (~real) | (flat & result["volume"].eq(0))
    return result


def resample_5m(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate on UTC 5-minute buckets, excluding filled bars from prices."""
    work = add_valid_mask(frame).sort_values(["root", "timestamp"], kind="stable")
    work["_bucket"] = work["timestamp"].dt.floor("5min")
    valid = ~work["is_filled"]
    for column in ("open", "high", "low", "close", "wap", "volume", "bar_count"):
        work[f"_valid_{column}"] = work[column].where(valid)
    work["_wap_volume"] = work["wap"].mul(work["volume"]).where(valid)
    grouped = work.groupby(["root", "_bucket"], sort=False, observed=True)

    result = grouped.agg(
        local_symbol=("local_symbol", "last"),
        contract_con_id=("contract_con_id", "last"),
        contract_expiry=("contract_expiry", "last"),
        open=("_valid_open", "first"),
        high=("_valid_high", "max"),
        low=("_valid_low", "min"),
        close=("_valid_close", "last"),
        volume=("_valid_volume", "sum"),
        bar_count=("_valid_bar_count", "sum"),
        wap_numerator=("_wap_volume", "sum"),
        n_valid_1m=("is_filled", lambda values: int((~values).sum())),
        fallback_close=("close", "last"),
    ).reset_index().rename(columns={"_bucket": "timestamp"})

    result["is_valid"] = result["n_valid_1m"].ge(2)
    result["wap"] = result["wap_numerator"].div(result["volume"].replace(0, np.nan))
    no_real = result["n_valid_1m"].eq(0)
    for column in ("open", "high", "low", "close", "wap"):
        result.loc[no_real, column] = np.nan
        result[column] = result.groupby("root", observed=True)[column].ffill()
        result[column] = result[column].fillna(result["fallback_close"])

    result["volume"] = result["volume"].fillna(0.0)
    result["bar_count"] = result["bar_count"].fillna(0).astype("int64")
    result["n_valid_1m"] = result["n_valid_1m"].astype("int8")
    return result[
        [
            "root", "timestamp", "local_symbol", "contract_con_id", "contract_expiry",
            "open", "high", "low", "close", "wap", "volume", "bar_count",
            "n_valid_1m", "is_valid",
        ]
    ]
