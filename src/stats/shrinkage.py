"""Second-order hierarchical empirical-Bayes shrinkage."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


DEFAULT_DIMENSIONS = ("structure", "efficiency", "volatility", "vol_direction", "liquidity", "phase")


def categorize_context(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "meta_branch" in result:
        structure = result["meta_branch"].astype(str)
        structure = structure.where(~structure.eq("none"), np.where(result["direction"] > 0, "long", "short"))
    elif "meta_event_type" in result:
        structure = result["meta_event_type"].astype(str) + "_" + np.where(result["direction"] > 0, "long", "short")
    else:
        structure = pd.Series(np.where(result["direction"] > 0, "long", "short"), index=result.index)
    result["structure"] = structure
    result["efficiency"] = pd.cut(result["er_20"], [-np.inf, 0.33, 0.66, np.inf], labels=["low", "mid", "high"]).astype(str)
    result["volatility"] = pd.cut(result["sigma_pct"], [-np.inf, 0.33, 0.66, np.inf], labels=["low", "mid", "high"]).astype(str)
    result["vol_direction"] = pd.cut(result["vol_ratio"], [-np.inf, 0.8, 1.2, np.inf], labels=["contract", "normal", "expand"]).astype(str)
    result["liquidity"] = pd.cut(result["liq_z"], [-np.inf, -1.0, 1.0, np.inf], labels=["low", "normal", "high"]).astype(str)
    result["phase"] = result["session_phase"].astype(str)
    return result


def _tau_and_weight(means: pd.Series, counts: pd.Series, sigma2: float) -> tuple[float, pd.Series]:
    sampling = sigma2 / counts.clip(lower=1)
    tau2 = max(float(means.var(ddof=1) - sampling.mean()), 0.0) if len(means) > 1 else 0.0
    if tau2 == 0 or not np.isfinite(tau2):
        return 0.0, pd.Series(0.0, index=means.index)
    weight = counts * tau2 / (counts * tau2 + sigma2)
    return tau2, weight.clip(0, 1)


def second_order_decomposition(
    frame: pd.DataFrame,
    outcome: str = "outcome",
    dimensions: tuple[str, ...] = DEFAULT_DIMENSIONS,
) -> tuple[pd.DataFrame, float]:
    data = frame.dropna(subset=[outcome, *dimensions]).copy()
    mu0 = float(data[outcome].mean()) if len(data) else np.nan
    sigma2 = float(data[outcome].var(ddof=1)) if len(data) > 1 else np.nan
    rows = []
    raw_main: dict[str, dict[str, float]] = {}
    for dimension in dimensions:
        grouped = data.groupby(dimension, observed=True)[outcome].agg(["count", "mean"])
        effects = grouped["mean"] - mu0
        tau2, weight = _tau_and_weight(effects, grouped["count"], sigma2)
        raw_main[dimension] = effects.to_dict()
        for level in grouped.index:
            rows.append({
                "term_order": 1, "variables": dimension, "level": str(level), "n": grouped.loc[level, "count"],
                "raw_effect": effects.loc[level], "posterior_effect": weight.loc[level] * effects.loc[level],
                "confidence": weight.loc[level], "tau2": tau2, "mu0": mu0,
            })
    for first, second in combinations(dimensions, 2):
        grouped = data.groupby([first, second], observed=True)[outcome].agg(["count", "mean"])
        interactions = pd.Series({
            level: row["mean"] - mu0 - raw_main[first].get(level[0], 0.0) - raw_main[second].get(level[1], 0.0)
            for level, row in grouped.iterrows()
        })
        tau2, weight = _tau_and_weight(interactions, grouped["count"], sigma2)
        for level in grouped.index:
            rows.append({
                "term_order": 2, "variables": f"{first}×{second}", "level": f"{level[0]}|{level[1]}",
                "n": grouped.loc[level, "count"], "raw_effect": interactions.loc[level],
                "posterior_effect": weight.loc[level] * interactions.loc[level], "confidence": weight.loc[level],
                "tau2": tau2, "mu0": mu0,
            })
    return pd.DataFrame(rows), sigma2


def root_hierarchy(frame: pd.DataFrame, outcome: str = "outcome") -> pd.DataFrame:
    data = frame.dropna(subset=[outcome, "root"])
    mu0 = float(data[outcome].mean()) if len(data) else np.nan
    sigma2 = float(data[outcome].var(ddof=1)) if len(data) > 1 else np.nan
    grouped = data.groupby("root", observed=True)[outcome].agg(["count", "mean"])
    effects = grouped["mean"] - mu0
    tau2, _ = _tau_and_weight(effects, grouped["count"], sigma2)
    # Cross-instrument deviations receive the strong prior required by the
    # research specification, even when noisy root means look dispersed.
    tau2 = min(tau2, sigma2 / 100) if np.isfinite(sigma2) else 0.0
    weight = grouped["count"] * tau2 / (grouped["count"] * tau2 + sigma2) if tau2 > 0 else pd.Series(0.0, index=grouped.index)
    output = grouped.reset_index().rename(columns={"mean": "raw_mean", "count": "n"})
    output["pooled_mean"] = mu0
    output["tau2_root"] = tau2
    output["confidence"] = output["root"].map(weight)
    output["posterior_mean"] = mu0 + output["confidence"] * (output["raw_mean"] - mu0)
    return output


def causal_cell_posterior(
    frame: pd.DataFrame,
    *,
    outcome: str = "outcome",
    available_time: str = "outcome_available_time",
    dimensions: tuple[str, ...] = DEFAULT_DIMENSIONS,
) -> pd.DataFrame:
    """Compute each trigger's posterior using only outcomes already observable."""
    ordered = frame.sort_values("timestamp", kind="stable").copy()
    observations = ordered.dropna(subset=[outcome, available_time, *dimensions]).sort_values(available_time)
    pending = list(observations.itertuples())
    pointer = 0
    cells: dict[tuple, list[float]] = {}
    global_n = 0; global_sum = 0.0; global_sum_squares = 0.0
    posterior_values = np.full(len(ordered), np.nan)
    confidence_values = np.zeros(len(ordered))
    prior_counts = np.zeros(len(ordered), dtype=np.int32)
    for position, row in enumerate(ordered.itertuples()):
        current_time = row.timestamp
        while pointer < len(pending) and getattr(pending[pointer], available_time) <= current_time:
            observed = pending[pointer]
            # Never let an observation inform its own decision at the same row.
            if observed.timestamp < current_time:
                key = tuple(getattr(observed, dimension) for dimension in dimensions)
                value = float(getattr(observed, outcome))
                state = cells.setdefault(key, [0, 0.0])
                state[0] += 1; state[1] += value
                global_n += 1; global_sum += value; global_sum_squares += value * value
            pointer += 1
        key = tuple(getattr(row, dimension) for dimension in dimensions)
        state = cells.get(key, [0, 0.0])
        if not global_n:
            continue
        mu0 = global_sum / global_n
        sigma2 = max((global_sum_squares - global_sum * global_sum / global_n) / (global_n - 1), 1e-12) if global_n > 1 else 1.0
        eligible = [values_ for values_ in cells.values() if values_[0] >= 2]
        means = np.array([values_[1] / values_[0] for values_ in eligible])
        counts = np.array([values_[0] for values_ in eligible])
        tau2 = max(float(np.var(means, ddof=1) - np.mean(sigma2 / counts)), 0.0) if len(means) > 1 else 0.0
        n = int(state[0]); prior_counts[position] = n
        if n and tau2 > 0:
            weight = n * tau2 / (n * tau2 + sigma2)
            posterior_values[position] = weight * (state[1] / n) + (1 - weight) * mu0
            confidence_values[position] = weight
        else:
            posterior_values[position] = mu0
    result = pd.DataFrame(
        {"posterior_mean": posterior_values, "confidence": confidence_values, "prior_cell_n": prior_counts},
        index=ordered.index,
    )
    return result.reindex(frame.index)
