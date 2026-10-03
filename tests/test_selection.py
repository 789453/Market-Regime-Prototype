import numpy as np
import pandas as pd

from src.selection import (
    add_plateau_diagnostics,
    economic_quality_weight,
    purged_anchored_splits,
    select_robust_candidate,
    summarize_walk_forward,
)


def test_economic_quality_separates_reliability_from_positive_edge():
    weights = economic_quality_weight(
        posterior_mean=np.array([-0.3, 0.0, 0.4]),
        posterior_se=np.array([0.1, 0.1, 0.1]),
        cost_in_outcome_units=0.05,
        reliability=np.array([0.9, 0.9, 0.9]),
    )
    assert weights[0] == 0
    assert weights[1] == 0
    assert weights[2] > 0.8


def test_purged_anchored_split_respects_embargo():
    timestamps = pd.date_range("2026-01-01", periods=1000, freq="5min", tz="UTC")
    splits = list(purged_anchored_splits(timestamps, n_splits=3, embargo_bars=24))
    assert len(splits) == 3
    for train, validation in splits:
        assert train.max() <= validation.min() - 25


def test_isolated_parameter_peak_is_not_selected():
    rows = []
    scores = {(1, 1): 0.01, (1, 2): 0.01, (2, 1): 0.01, (2, 2): 0.10, (2, 3): 0.01, (3, 2): 0.01}
    for (a, b), score in scores.items():
        for fold in range(4):
            for root in ("ES", "NQ", "RTY"):
                for _ in range(15):
                    rows.append({"a": a, "b": b, "fold": fold, "root": root, "net_outcome": score})
    summary = summarize_walk_forward(pd.DataFrame(rows), ["a", "b"], minimum_triggers=100)
    summary = add_plateau_diagnostics(summary, ["a", "b"])
    assert not summary.loc[(summary.a == 2) & (summary.b == 2), "robust_pass"].iloc[0]
    selected = select_robust_candidate(summary)
    assert selected is not None
    assert (selected.a, selected.b) != (2, 2)
