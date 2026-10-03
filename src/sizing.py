"""Volatility-targeted, cost-aware position sizing for Stage 5."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


BARS_PER_SESSION = {"ES": 189, "NQ": 189, "RTY": 189, "HSI": 54, "HTI": 54}


def size_instruments(
    fused: pd.DataFrame,
    context: pd.DataFrame,
    *,
    instrument_vol_target: float = 0.10,
    eta: float = 0.25,
    max_weight: float = 2.0,
    max_single_bet_fraction: float = 0.25,
    portfolio_vol_target: float = 0.10,
    breaker_multiplier: float = 0.30,
    sigma_percentile_breaker: float = 0.99,
    spread_percentile_breaker: float = 0.95,
    band_zeta: float = 0.05,
    risk_aversion: float = 1.0,
) -> pd.DataFrame:
    """Create bounded targets and causal no-trade bands on the signal index."""
    ctx = context.reindex(fused.index)
    roots = fused.index.get_level_values("root")
    annual_bars = np.array([252 * BARS_PER_SESSION.get(root, 189) for root in roots], dtype=float)
    sigma = ctx["sigma_hat"].to_numpy(dtype=float)
    sigma_ann = sigma * np.sqrt(annual_bars)
    raw = instrument_vol_target / np.where(sigma_ann > 0, sigma_ann, np.inf)
    raw *= np.tanh(fused["mu"].to_numpy(dtype=float) / eta)
    raw = np.clip(raw, -max_weight, max_weight)

    horizon = fused.get("active_horizon", pd.Series(1.0, index=fused.index)).to_numpy(dtype=float)
    horizon = np.maximum(horizon, 1.0)
    per_bet_budget = max_single_bet_fraction * portfolio_vol_target / np.sqrt(252.0)
    bet_cap = per_bet_budget / np.maximum(sigma * np.sqrt(horizon), 1e-12)
    raw = np.clip(raw, -bet_cap, bet_cap)

    sigma_break = ctx["sigma_pct"].fillna(0.0).to_numpy(dtype=float) > sigma_percentile_breaker
    spread = ctx["spread_bp"].astype(float)
    spread_limit = pd.Series(np.inf, index=fused.index, dtype=float)
    for _, loc in spread.groupby(level="root", sort=False).groups.items():
        values = spread.loc[loc]
        causal_q = values.expanding(min_periods=100).quantile(spread_percentile_breaker).shift(1)
        spread_limit.loc[loc] = causal_q.fillna(np.inf)
    spread_break = spread.to_numpy() > spread_limit.to_numpy()
    breaker = sigma_break | spread_break
    target = raw * np.where(breaker, breaker_multiplier, 1.0)

    cost = ctx.get("round_trip_cost_bp", spread).fillna(spread).clip(lower=0.0).to_numpy() * 1e-4
    band = band_zeta * np.sqrt(
        cost / np.maximum(risk_aversion * np.square(sigma) * horizon, 1e-12)
    )
    band = np.minimum(band, max_weight)
    return pd.DataFrame(
        {
            "mu": fused["mu"],
            "target_unscaled": target,
            "target": target,
            "band": band,
            "sigma_ann": sigma_ann,
            "breaker": breaker,
            "active_horizon": horizon,
            "signal_active": fused.get("confidence_mass", pd.Series(0.0, index=fused.index)).to_numpy() > 0,
        },
        index=fused.index,
    )


def causal_daily_correlations(
    bars: pd.DataFrame,
    context: pd.DataFrame,
    roots: tuple[str, ...] = ("ES", "NQ", "RTY", "HSI"),
    half_life_sessions: float = 20.0,
    lookback_days: int = 60,
) -> dict[object, np.ndarray]:
    """Prior-day exponentially weighted Ledoit-Wolf correlation matrices."""
    idx = bars.index.intersection(context.index)
    work = pd.DataFrame(index=idx)
    work["z"] = bars.loc[idx, "ret"] / context.loc[idx, "sigma_hat"].replace(0, np.nan)
    work = work.loc[work.index.get_level_values("root").isin(roots)]
    pivot = work["z"].unstack("root").reindex(columns=roots)
    dates = pd.Index(pivot.index.date, name="date")
    unique_dates = pd.Index(pd.unique(dates)).sort_values()
    result: dict[object, np.ndarray] = {}
    identity = np.eye(len(roots))
    for pos, date in enumerate(unique_dates):
        previous = unique_dates[max(0, pos - lookback_days) : pos]
        if len(previous) < 5:
            result[date] = identity.copy()
            continue
        mask = dates.isin(previous)
        sample = pivot.loc[mask].dropna(how="any")
        if len(sample) < 100:
            result[date] = identity.copy()
            continue
        age_map = {d: len(previous) - 1 - i for i, d in enumerate(previous)}
        age = np.array([age_map[d] for d in sample.index.date], dtype=float)
        weights = np.power(0.5, age / half_life_sessions)
        x = sample.to_numpy(dtype=float)
        x = x - np.average(x, axis=0, weights=weights)
        x *= np.sqrt(weights / weights.mean())[:, None]
        cov = LedoitWolf(assume_centered=True).fit(x).covariance_
        scale = np.sqrt(np.maximum(np.diag(cov), 1e-12))
        corr = cov / np.outer(scale, scale)
        result[date] = np.clip(corr, -1.0, 1.0)
    return result


def scale_portfolio_targets(
    sized: pd.DataFrame,
    correlations: dict[object, np.ndarray],
    *,
    roots: tuple[str, ...] = ("ES", "NQ", "RTY", "HSI"),
    portfolio_vol_target: float = 0.10,
    max_weight: float = 2.0,
    same_direction_multiple: float = 2.5,
) -> pd.DataFrame:
    """Shrink cross-asset targets to the portfolio volatility and concentration caps."""
    out = sized.copy()
    target = out["target"].unstack("root").reindex(columns=roots).fillna(0.0)
    vol = out["sigma_ann"].unstack("root").reindex(columns=roots)
    scaled = target.copy()
    for i, timestamp in enumerate(target.index):
        w = target.iloc[i].to_numpy(dtype=float)
        sig = vol.iloc[i].fillna(0.0).to_numpy(dtype=float)
        corr = correlations.get(timestamp.date(), np.eye(len(roots)))
        covariance = np.outer(sig, sig) * corr
        forecast = float(np.sqrt(max(w @ covariance @ w, 0.0)))
        factor = min(1.0, portfolio_vol_target / forecast) if forecast > 0 else 1.0
        w *= factor
        positive = w > 0
        negative = w < 0
        cap = same_direction_multiple * max_weight
        if w[positive].sum() > cap:
            w[positive] *= cap / w[positive].sum()
        if -w[negative].sum() > cap:
            w[negative] *= cap / (-w[negative].sum())
        scaled.iloc[i] = np.clip(w, -max_weight, max_weight)
    stacked = scaled.stack(future_stack=True).rename("target")
    stacked.index.names = ["timestamp", "root"]
    stacked = stacked.reorder_levels(["root", "timestamp"]).sort_index()
    out["target"] = stacked.reindex(out.index).fillna(0.0)
    return out
