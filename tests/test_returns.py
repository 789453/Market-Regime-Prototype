import numpy as np
import pandas as pd

from src.returns import add_returns


def test_roll_boundary_nan() -> None:
    frame = pd.DataFrame(
        {
            "root": ["ES"] * 5,
            "timestamp": pd.date_range("2026-03-09 13:30:00Z", periods=5, freq="5min"),
            "close": [100.0, 101.0, 110.0, 111.0, 112.0],
            "is_valid": [True, True, True, False, True],
            "local_symbol": ["ESH6", "ESH6", "ESM6", "ESM6", "ESM6"],
            "session_id": pd.to_datetime(["2026-03-09"] * 5).date,
        }
    )

    result = add_returns(frame)

    assert not bool(result.loc[0, "is_roll_boundary"])
    assert bool(result.loc[2, "is_roll_boundary"])
    assert np.isnan(result.loc[0, "ret"])
    assert np.isclose(result.loc[1, "ret"], np.log(101.0 / 100.0))
    assert np.isnan(result.loc[2, "ret"])
    assert np.isnan(result.loc[3, "ret"])
    assert np.isnan(result.loc[4, "ret"])


def test_cross_session_is_overnight_not_intraday() -> None:
    frame = pd.DataFrame(
        {
            "root": ["ES", "ES"],
            "timestamp": pd.to_datetime(["2026-03-09 20:55Z", "2026-03-09 22:00Z"]),
            "close": [100.0, 101.0],
            "is_valid": [True, True],
            "local_symbol": ["ESH6", "ESH6"],
            "session_id": pd.to_datetime(["2026-03-09", "2026-03-10"]).date,
        }
    )

    result = add_returns(frame)

    assert np.isnan(result.loc[1, "ret"])
    assert np.isclose(result.loc[1, "overnight_ret"], np.log(101.0 / 100.0))

