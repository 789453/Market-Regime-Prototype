import numpy as np
import pandas as pd

from src.forensics import infer_tick_sizes, roll_boundary_table
from src.spread import estimate_chl_spread


def test_tick_inference_uses_ohlc_grid_not_wap() -> None:
    frame = pd.DataFrame(
        {
            "root": ["ES"] * 3 + ["RTY"] * 3,
            "open": [100.0, 100.25, 100.5, 200.0, 200.1, 200.2],
            "high": [100.25, 100.5, 100.75, 200.1, 200.2, 200.3],
            "low": [99.75, 100.0, 100.25, 199.9, 200.0, 200.1],
            "close": [100.25, 100.5, 100.75, 200.1, 200.2, 200.3],
        }
    )
    ticks = infer_tick_sizes(frame).set_index("root")["tick_size"]
    assert np.isclose(ticks["ES"], 0.25)
    assert np.isclose(ticks["RTY"], 0.1)


def test_roll_table_records_previous_and_next_contract() -> None:
    frame = pd.DataFrame(
        {
            "root": ["ES"] * 3,
            "timestamp": pd.to_datetime(["2026-03-01", "2026-03-02", "2026-03-04"], utc=True),
            "local_symbol": ["ESH6", "ESH6", "ESM6"],
            "open": [100.0, 101.0, 103.0],
            "close": [100.0, 101.0, 103.0],
        }
    )
    result = roll_boundary_table(frame)
    assert len(result) == 1
    assert result.loc[0, "previous_symbol"] == "ESH6"
    assert result.loc[0, "next_symbol"] == "ESM6"
    assert result.loc[0, "gap_minutes"] == 2 * 24 * 60


def test_chl_spread_respects_one_tick_floor() -> None:
    timestamps = pd.date_range("2026-03-09 13:30Z", periods=4, freq="5min")
    bars = pd.DataFrame(
        {
            "root": ["ES"] * 4,
            "timestamp": timestamps,
            "session_id": [pd.Timestamp("2026-03-09").date()] * 4,
            "local_minute": [570, 575, 580, 585],
            "close": [100.0, 100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 100.0, 100.0],
            "low": [100.0, 100.0, 100.0, 100.0],
            "is_valid": [True] * 4,
        }
    )
    ticks = pd.DataFrame({"root": ["ES"], "tick_size": [0.25], "median_close": [100.0], "tick_bp": [25.0]})
    spread = estimate_chl_spread(bars, ticks)
    assert (spread["spread_bp"] >= 25.0).all()
