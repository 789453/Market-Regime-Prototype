"""Verify event timing primitives and a genuine non-rebalanced benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.crypto_event_research import _future_side_variance, add_hysteresis, evaluate_events


def test_future_side_variance_excludes_entry_and_uses_only_horizon() -> None:
    increments = np.r_[0.0, np.tile([0.01, -0.02], 30)]
    log_open = np.cumsum(increments)
    up, down = _future_side_variance(log_open, np.array([0, 1, 13]))
    assert np.isclose(up[0], 24 * 0.01**2)
    assert np.isclose(down[0], 24 * 0.02**2)
    assert np.isnan(up[-1])


def test_hysteresis_holds_through_neutral_and_exits_on_sign_change() -> None:
    frame = pd.DataFrame({
        "symbol": ["BTCUSDT"] * 4,
        "execution_at": pd.date_range("2024-01-01", periods=4, freq="15min", tz="UTC"),
        "pred_logret": [0.003, 0.001, -0.0001, -0.003],
    })
    result = add_hysteresis(frame)
    assert result.hysteresis_position.tolist() == [1.0, 1.0, 0.0, -1.0]


def test_buyhold_does_not_rebalance_each_bar() -> None:
    times = pd.date_range("2024-01-01", periods=3, freq="15min", tz="UTC")
    frame = pd.DataFrame({
        "symbol": ["A"] * 3 + ["B"] * 3,
        "execution_at": list(times) * 2,
        "execution_price": [100.0, 200.0, 100.0, 100.0, 50.0, 100.0],
        "target_position": [0.0] * 6,
        "hysteresis_position": [0.0] * 6,
    })
    score, _ = evaluate_events(frame)
    hold = score.loc[(score.strategy == "buyhold") & (score.side_cost_bp == 0)].iloc[0]
    assert np.isclose(hold.cumulative_return, 0.0)
