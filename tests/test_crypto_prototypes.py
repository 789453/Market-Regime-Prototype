"""Prototype geometry and event sparsification contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.crypto.prototypes import Coordinate, Prototype, QuantileReference, similarity, sparse_events


def test_similarity_has_flat_core_and_directional_penalty() -> None:
    proto = Prototype("up", +1, "test", {"trend": Coordinate(.8, .1, 2), "flow": Coordinate(.7, .1, 1)})
    values = pd.DataFrame({"trend": [.8, .75, .1, .8], "flow": [.7, .75, .7, np.nan]})
    score = similarity(values, proto)
    assert np.isclose(score[0], 1)
    assert .98 < score[1] < score[0]
    assert score[2] < .1
    assert np.isnan(score[3])


def test_quantile_reference_is_fixed_after_training() -> None:
    training = pd.DataFrame({"value": np.arange(100.0)})
    ref = QuantileReference(training, ["value"])
    before = ref.transform(pd.DataFrame({"value": [50.0]})).value.iloc[0]
    # Later market observations do not refit the reference distribution.
    after = ref.transform(pd.DataFrame({"value": [50.0, 10_000.0]})).value.iloc[0]
    assert before == after


def test_sparse_events_uses_per_symbol_refractory_period() -> None:
    frame = pd.DataFrame({
        "symbol": ["A", "B", "A", "A", "B"],
        "available_at": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 05:00", "2024-01-01 05:00"], utc=True),
    })
    selected = sparse_events(frame, np.ones(len(frame), dtype=bool), min_gap_hours=4)
    assert len(selected) == 4
    assert set(selected.loc[selected.symbol == "A", "available_at"].dt.hour) == {0, 5}
