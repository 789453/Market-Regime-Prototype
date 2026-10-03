"""Contracts for all-field historical geometry and path construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.crypto.full_patterns import BASE_COLUMNS, MODEL_COLUMNS, EmpiricalMap, add_path


def test_full_representation_covers_every_dual_frequency_field() -> None:
    assert len(BASE_COLUMNS) == 112
    assert len(MODEL_COLUMNS) == 136
    assert len(set(MODEL_COLUMNS)) == len(MODEL_COLUMNS)


def test_history_path_is_per_symbol_and_prefix_causal() -> None:
    base = pd.DataFrame({
        "symbol": ["A"] * 20 + ["B"] * 20,
        "available_at": list(pd.date_range("2024-01-01", periods=20, freq="15min", tz="UTC")) * 2,
        "rv_fast_slow_15m": np.r_[np.arange(20.0), np.arange(100.0, 120.0)],
    })
    # Other path representatives are present but need no special values here.
    from src.crypto.full_patterns import PATH_REPRESENTATIVES

    for names in PATH_REPRESENTATIVES.values():
        for name in names:
            if name not in base:
                base[name] = 0.0
    whole = add_path(base)
    prefix = add_path(base.loc[base.available_at < pd.Timestamp("2024-01-01 04:30", tz="UTC")])
    column = "rv_fast_slow_15m__lag4h"
    assert whole.loc[whole.symbol.eq("B"), column].iloc[16] == 100.0
    assert whole.loc[whole.symbol.eq("A"), column].iloc[16] == 0.0
    pd.testing.assert_series_equal(whole.loc[prefix.index, column], prefix[column])


def test_empirical_map_is_training_only_and_handles_missing() -> None:
    reference = EmpiricalMap(pd.DataFrame({"x": np.arange(100.0), "constant": np.ones(100)}), ("x", "constant"))
    mapped, missing = reference.transform(pd.DataFrame({"x": [50.0, np.nan], "constant": [1.0, 1.0]}))
    assert .45 < mapped[0, 0] < .55
    assert mapped[1, 0] == .5 and missing[1, 0]
    assert mapped[0, 1] == .5
    later, _ = reference.transform(pd.DataFrame({"x": [50.0, 1e9], "constant": [1.0, 1.0]}))
    assert later[0, 0] == mapped[0, 0]


def test_zero_inflated_ties_get_midrank_not_false_extreme() -> None:
    training = pd.DataFrame({"flag": [0.0] * 95 + [1.0] * 5})
    mapped, _ = EmpiricalMap(training, ("flag",)).transform(pd.DataFrame({"flag": [0.0, 1.0]}))
    assert .4 < mapped[0, 0] < .55
    assert mapped[1, 0] > mapped[0, 0]
