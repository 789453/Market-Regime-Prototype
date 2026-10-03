"""Matched random-timing null distributions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def matched_null_distribution(
    analysis: pd.DataFrame,
    horizon: int,
    *,
    draws: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    """Circularly shift each root×phase trigger pattern over valid candidates.

    The operation exactly preserves trigger count, direction sequence and
    within-stratum clustering distances while matching the phase distribution.
    """
    rng = np.random.default_rng(seed)
    sums = np.zeros(draws)
    counts = np.zeros(draws, dtype=int)
    outcome_column = f"fwd_ret_z_{horizon}"
    valid_column = f"label_valid_{horizon}"
    work = analysis.loc[analysis[valid_column] & analysis[outcome_column].notna()].copy()
    for _, group in work.groupby(["root", "session_phase"], sort=False, observed=True):
        trigger_mask = group["direction"].ne(0).to_numpy()
        if not trigger_mask.any():
            continue
        outcomes = group[outcome_column].to_numpy(dtype=float)
        positions = np.flatnonzero(trigger_mask)
        directions = group.loc[group["direction"].ne(0), "direction"].to_numpy(dtype=float)
        for start in range(0, draws, 100):
            stop = min(start + 100, draws)
            offsets = rng.integers(0, len(group), size=stop - start)
            sampled = outcomes[(positions[None, :] + offsets[:, None]) % len(group)]
            sums[start:stop] += (sampled * directions).sum(axis=1)
            counts[start:stop] += len(positions)
    return np.divide(sums, counts, out=np.full(draws, np.nan), where=counts > 0)


def null_percentile(actual: float, distribution: np.ndarray) -> float:
    valid = distribution[np.isfinite(distribution)]
    return float(np.mean(valid <= actual)) if len(valid) else np.nan
