"""Low-dimensional, observable, causal context engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.spread import causal_chl_spread


def _session_percentile(frame: pd.DataFrame, window: int = 60) -> pd.Series:
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    for root, group in frame.groupby("root", sort=False, observed=True):
        session_values = group.groupby("session_id", sort=False)["sigma_day"].first()
        percentiles = {}
        for i, (session_id, current) in enumerate(session_values.items()):
            history = session_values.iloc[max(0, i - window) : i].dropna()
            percentiles[session_id] = float((history <= current).mean()) if len(history) >= 5 and np.isfinite(current) else np.nan
        output.loc[group.index] = group["session_id"].map(percentiles)
    return output


def _efficiency_ratio(returns: pd.Series, roots: pd.Series, window: int) -> pd.Series:
    numerator = returns.groupby(roots, sort=False).transform(
        lambda values: values.rolling(window, min_periods=window).sum().abs()
    )
    denominator = returns.abs().groupby(roots, sort=False).transform(
        lambda values: values.rolling(window, min_periods=window).sum()
    )
    return numerator.div(denominator.replace(0, np.nan)).clip(0, 1)


def _variance_ratio(returns: pd.Series, roots: pd.Series, q: int = 5, window: int = 120) -> pd.Series:
    output = pd.Series(np.nan, index=returns.index, dtype=float)
    for root, index in roots.groupby(roots, sort=False).groups.items():
        one = returns.loc[index]
        q_return = one.rolling(q, min_periods=q).sum()
        numerator = q_return.rolling(window, min_periods=window).var()
        denominator = q * one.rolling(window, min_periods=window).var()
        output.loc[index] = numerator.div(denominator.replace(0, np.nan))
    return output


def _causal_liquidity_z(frame: pd.DataFrame) -> pd.Series:
    log_volume = np.log1p(frame["volume"])
    keys = [frame["root"], frame["session_phase"]]
    median = log_volume.groupby(keys, sort=False).transform(
        lambda values: values.expanding(min_periods=20).median().shift(1)
    )
    deviation = (log_volume - median).abs()
    mad = deviation.groupby(keys, sort=False).transform(
        lambda values: values.expanding(min_periods=20).median().shift(1)
    )
    return ((log_volume - median) / (1.4826 * mad).replace(0, np.nan)).clip(-8, 8)


def build_context(bars: pd.DataFrame, ticks: pd.DataFrame) -> pd.DataFrame:
    result = bars.sort_values(["root", "timestamp"], kind="stable").copy()
    standardized = result["ret"] / result["sigma_hat"]
    squared = standardized.pow(2)
    fast = squared.groupby(result["root"], sort=False).transform(
        lambda values: values.ewm(halflife=12, adjust=False, min_periods=12).mean()
    )
    slow = squared.groupby(result["root"], sort=False).transform(
        lambda values: values.ewm(halflife=96, adjust=False, min_periods=24).mean()
    )
    result["sigma_pct"] = _session_percentile(result)
    result["vol_ratio"] = fast.div(slow.replace(0, np.nan))
    result["er_20"] = _efficiency_ratio(result["ret"], result["root"], 20)
    result["er_60"] = _efficiency_ratio(result["ret"], result["root"], 60)
    result["vr_5"] = _variance_ratio(standardized, result["root"])
    result["liq_z"] = _causal_liquidity_z(result)
    base_spread = causal_chl_spread(result, ticks)
    result["spread_bp"] = base_spread * (1 + 0.25 * (-result["liq_z"]).clip(lower=0).fillna(0))
    result["avg_trade_size"] = result["volume"].div(result["bar_count"].replace(0, np.nan))

    es = result.loc[result["root"].eq("ES"), ["timestamp"]].copy()
    es["es_ret_z"] = standardized.loc[es.index].to_numpy()
    es_map = es.set_index("timestamp")["es_ret_z"]
    result["es_ret_z"] = result["timestamp"].map(es_map)

    session_date = pd.to_datetime(result["session_id"])
    expiry = pd.to_datetime(result["contract_expiry"].astype(str), format="%Y%m%d")
    days_to_expiry = (expiry - session_date).dt.days
    result["is_roll_week"] = days_to_expiry.between(0, 7)
    result["is_month_end"] = (session_date + pd.offsets.BDay(2)).dt.month.ne(session_date.dt.month)
    result["is_quarter_end"] = result["is_month_end"] & session_date.dt.month.isin([3, 6, 9, 12])

    tick_bp = result["root"].map(ticks.set_index("root")["tick_bp"])
    execution_components = pd.concat(
        [
            np.tanh(result["liq_z"] / 2),
            -np.tanh(np.log(result["vol_ratio"].clip(lower=1e-6))),
            1 - 2 * result["sigma_pct"],
            -np.tanh(np.log(result["spread_bp"] / tick_bp)),
        ],
        axis=1,
    ).fillna(0)
    score = execution_components.mean(axis=1)
    result["gate_score"] = 0.2 + 1.3 / (1 + np.exp(-score))

    columns = [
        "root", "timestamp", "sigma_hat", "sigma_day", "seas", "u", "sigma_pct",
        "vol_ratio", "er_20", "er_60", "vr_5", "liq_z", "spread_bp",
        "avg_trade_size", "es_ret_z", "gate_score", "session_phase", "phi",
        "is_roll_week", "is_month_end", "is_quarter_end", "is_half_day",
    ]
    return result[columns]
