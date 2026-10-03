"""Standard detector statistical cards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def stationary_bootstrap_mean(values: np.ndarray, block_length: int, draws: int, seed: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.full(draws, np.nan)
    rng = np.random.default_rng(seed)
    restart_probability = 1 / max(block_length, 1)
    indices = np.empty((draws, len(values)), dtype=np.int32)
    indices[:, 0] = rng.integers(len(values), size=draws)
    for position in range(1, len(values)):
        restart = rng.random(draws) < restart_probability
        fresh = rng.integers(len(values), size=draws)
        indices[:, position] = np.where(restart, fresh, (indices[:, position - 1] + 1) % len(values))
    return values[indices].mean(axis=1)


def _newey_west_t(values: np.ndarray, lag: int) -> float:
    if len(values) < max(10, lag + 2):
        return np.nan
    model = sm.OLS(values, np.ones((len(values), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    return float(model.tvalues[0])


def build_stat_card(analysis: pd.DataFrame, detector_id: str, horizons: list[int], seed: int, bootstrap_draws: int = 500):
    trigger = analysis.loc[analysis["direction"].ne(0)].copy()
    summary = []
    for horizon in horizons:
        valid = trigger.loc[trigger[f"label_valid_{horizon}"]].dropna(subset=[f"fwd_ret_z_{horizon}", f"fwd_ret_{horizon}"])
        signed_z = (valid["direction"] * valid[f"fwd_ret_z_{horizon}"]).to_numpy()
        signed_raw = valid["direction"] * valid[f"fwd_ret_{horizon}"]
        boot = stationary_bootstrap_mean(signed_z, 5 * horizon, bootstrap_draws, seed + horizon)
        standard_error = np.nanstd(boot, ddof=1)
        mean_z = float(np.nanmean(signed_z)) if len(signed_z) else np.nan
        wins = signed_z[signed_z > 0]
        losses = signed_z[signed_z < 0]
        summary.append({
            "detector_id": detector_id, "horizon": horizon, "n": len(valid),
            "mean_z": mean_z, "ci_low": float(np.nanquantile(boot, 0.025)), "ci_high": float(np.nanquantile(boot, 0.975)),
            "raw_t": mean_z / (np.nanstd(signed_z, ddof=1) / np.sqrt(len(signed_z))) if len(signed_z) > 1 else np.nan,
            "nw_t": _newey_west_t(signed_z, horizon),
            "stationary_t": mean_z / standard_error if standard_error > 0 else np.nan,
            "event_sharpe": mean_z / np.nanstd(signed_z, ddof=1) if len(signed_z) > 1 else np.nan,
            "win_rate": float(np.mean(signed_z > 0)) if len(signed_z) else np.nan,
            "payoff_ratio": float(np.mean(wins) / abs(np.mean(losses))) if len(wins) and len(losses) else np.nan,
            "mean_raw": float(signed_raw.mean()) if len(valid) else np.nan,
            "mean_net": float((signed_raw - valid["cost_return"]).mean()) if len(valid) else np.nan,
        })
    breakdowns = {}
    default_horizon = int(trigger["horizon"].mode().iloc[0]) if len(trigger) else horizons[0]
    valid = trigger.loc[trigger[f"label_valid_{default_horizon}"]].copy()
    valid["signed_z"] = valid["direction"] * valid[f"fwd_ret_z_{default_horizon}"]
    for name, columns in {
        "root": ["root"], "phase": ["session_phase"], "month": ["month"], "side": ["direction"]
    }.items():
        breakdowns[name] = valid.groupby(columns, observed=True)["signed_z"].agg(["count", "mean", "std"]).reset_index()
    if len(valid) >= 3:
        valid["magnitude_bin"] = pd.qcut(valid["magnitude"].rank(method="first"), 3, labels=["low", "mid", "high"])
        breakdowns["magnitude"] = valid.groupby("magnitude_bin", observed=True)["signed_z"].agg(["count", "mean", "std"]).reset_index()
    else:
        breakdowns["magnitude"] = pd.DataFrame()
    intervals = trigger.sort_values(["root", "timestamp"]).copy()
    intervals["interval_hours"] = intervals.groupby("root", observed=True)["timestamp"].diff() / pd.Timedelta(hours=1)
    breakdowns["interval"] = intervals.groupby("root", observed=True)["interval_hours"].quantile([0.1, 0.5, 0.9]).unstack().reset_index().rename(columns={0.1: "p10", 0.5: "median", 0.9: "p90"})
    return pd.DataFrame(summary), breakdowns
