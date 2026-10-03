"""Bias-aware candidate selection utilities for the post-Stage-5 research loop.

This module deliberately does not inspect the reserved holdout.  It separates
estimation reliability from the probability of a positive net economic edge,
and selects parameter plateaus across purged anchored folds rather than peaks.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm


def economic_quality_weight(
    posterior_mean: pd.Series | np.ndarray,
    posterior_se: pd.Series | np.ndarray,
    cost_in_outcome_units: pd.Series | np.ndarray | float,
    reliability: pd.Series | np.ndarray,
    *,
    prior_qualified: pd.Series | np.ndarray | bool = True,
    minimum_probability: float = 0.60,
) -> np.ndarray:
    """Causal fusion weight for a preregistered direction after estimated cost.

    Reliability answers how strongly a cell is estimated.  Economic quality
    answers whether its signed edge clears cost.  A negative edge is shrunk to
    zero; the function never flips a detector direction after observing data.
    """
    mean = np.asarray(posterior_mean, dtype=float)
    se = np.maximum(np.asarray(posterior_se, dtype=float), 1e-12)
    cost = np.asarray(cost_in_outcome_units, dtype=float)
    reliable = np.clip(np.asarray(reliability, dtype=float), 0.0, 1.0)
    qualified = np.asarray(prior_qualified, dtype=bool)
    probability = norm.cdf((mean - cost) / se)
    quality = np.clip((probability - minimum_probability) / (1.0 - minimum_probability), 0.0, 1.0)
    return reliable * quality * qualified


def purged_anchored_splits(
    timestamps: Sequence,
    *,
    n_splits: int = 4,
    minimum_train_fraction: float = 0.40,
    embargo_bars: int = 96,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield expanding-train, forward-validation indices with a fixed purge."""
    times = pd.Index(pd.to_datetime(timestamps, utc=True))
    order = np.argsort(times.asi8, kind="stable")
    n = len(order)
    first_validation = int(np.ceil(n * minimum_train_fraction))
    validation_size = max((n - first_validation) // n_splits, 1)
    for split in range(n_splits):
        validation_start = first_validation + split * validation_size
        validation_stop = n if split == n_splits - 1 else min(n, validation_start + validation_size)
        train_stop = max(validation_start - embargo_bars, 0)
        if train_stop == 0 or validation_start >= validation_stop:
            continue
        yield order[:train_stop], order[validation_start:validation_stop]


def summarize_walk_forward(
    results: pd.DataFrame,
    candidate_columns: Sequence[str],
    *,
    outcome: str = "net_outcome",
    minimum_triggers: int = 150,
) -> pd.DataFrame:
    """Summarize fold/root consistency without rewarding one large fold."""
    required = [*candidate_columns, "fold", "root", outcome]
    data = results.dropna(subset=required)
    rows = []
    group_key = candidate_columns[0] if len(candidate_columns) == 1 else list(candidate_columns)
    for key, frame in data.groupby(group_key, sort=False):
        key = (key,) if len(candidate_columns) == 1 else tuple(key)
        fold_means = frame.groupby("fold")[outcome].mean()
        root_means = frame.groupby("root")[outcome].mean()
        row = dict(zip(candidate_columns, key))
        row.update(
            {
                "triggers": len(frame),
                "mean_net": frame[outcome].mean(),
                "median_fold_net": fold_means.median(),
                "positive_fold_rate": (fold_means > 0).mean(),
                "positive_roots": int((root_means > 0).sum()),
                "worst_root_net": root_means.min(),
                "fold_dispersion": fold_means.std(ddof=1),
            }
        )
        row["base_pass"] = bool(
            row["triggers"] >= minimum_triggers
            and row["median_fold_net"] > 0
            and row["positive_fold_rate"] >= 0.60
            and row["positive_roots"] >= 2
        )
        rows.append(row)
    return pd.DataFrame(rows)


def add_plateau_diagnostics(
    summary: pd.DataFrame,
    parameter_columns: Sequence[str],
    *,
    score: str = "median_fold_net",
    relative_floor: float = 0.80,
) -> pd.DataFrame:
    """Require adjacent grid points to share most of a candidate's performance."""
    output = summary.copy()
    level_maps = {
        column: {value: rank for rank, value in enumerate(sorted(output[column].dropna().unique()))}
        for column in parameter_columns
    }
    plateau_rates = []
    neighbor_counts = []
    for _, row in output.iterrows():
        distances = np.zeros(len(output), dtype=int)
        for column in parameter_columns:
            ranks = output[column].map(level_maps[column]).to_numpy()
            distances += np.abs(ranks - level_maps[column][row[column]])
        neighbors = output.loc[distances == 1]
        neighbor_counts.append(len(neighbors))
        if len(neighbors) == 0 or row[score] <= 0:
            plateau_rates.append(0.0)
        else:
            plateau_rates.append(float((neighbors[score] >= relative_floor * row[score]).mean()))
    output["neighbor_count"] = neighbor_counts
    output["plateau_rate"] = plateau_rates
    output["robust_pass"] = output["base_pass"] & (output["neighbor_count"] >= 2) & (output["plateau_rate"] >= 0.50)
    return output


def select_robust_candidate(summary: pd.DataFrame) -> pd.Series | None:
    """Select by fold median only among candidates that passed the plateau rules."""
    eligible = summary.loc[summary["robust_pass"]]
    if eligible.empty:
        return None
    return eligible.sort_values(
        ["median_fold_net", "positive_fold_rate", "worst_root_net"], ascending=False
    ).iloc[0]
