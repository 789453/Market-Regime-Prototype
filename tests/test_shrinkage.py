import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.stats.shrinkage import causal_cell_posterior, second_order_decomposition


DIMENSIONS = ("structure", "efficiency", "volatility", "vol_direction", "liquidity", "phase")


def _frame(n=80):
    timestamp = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    frame = pd.DataFrame({
        "timestamp": timestamp,
        "outcome_available_time": timestamp + pd.Timedelta(minutes=30),
        "outcome": np.sin(np.arange(n) / 7),
        "structure": np.where(np.arange(n) % 2, "long", "short"),
        "efficiency": np.where(np.arange(n) % 3, "mid", "high"),
        "volatility": "mid", "vol_direction": "normal", "liquidity": "normal", "phase": "ASIA",
    })
    return frame


def test_causal_cell_posterior_is_prefix_invariant():
    frame = _frame()
    full = causal_cell_posterior(frame)
    partial = causal_cell_posterior(frame.iloc[:55].copy())
    assert_frame_equal(partial, full.iloc[:55])


def test_hierarchical_model_stops_at_second_order():
    effects, _ = second_order_decomposition(_frame())
    assert effects["term_order"].max() == 2
    assert effects["term_order"].min() == 1
