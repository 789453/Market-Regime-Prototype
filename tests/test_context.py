import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.context import build_context


def _bars(n: int = 180) -> pd.DataFrame:
    timestamps = pd.date_range("2026-03-02 23:00Z", periods=n, freq="5min")
    ret = pd.Series(np.sin(np.arange(n) / 9) * 0.001)
    close = 100 * np.exp(ret.cumsum())
    return pd.DataFrame(
        {
            "root": "ES",
            "timestamp": timestamps,
            "session_id": [pd.Timestamp("2026-03-03").date()] * n,
            "session_phase": ["ASIA"] * n,
            "local_minute": (18 * 60 + np.arange(n) * 5) % (24 * 60),
            "phi": np.arange(n) / 275,
            "local_symbol": ["ESH6"] * n,
            "contract_expiry": [20260320] * n,
            "ret": ret,
            "close": close,
            "high": close * 1.0001,
            "low": close * 0.9999,
            "volume": np.arange(n) + 100.0,
            "bar_count": np.arange(n) + 20,
            "is_valid": True,
            "is_half_day": False,
            "sigma_day": 0.001,
            "seas": 1.0,
            "sigma_hat": 0.001,
            "u": 1.0,
        }
    )


def test_context_is_prefix_invariant() -> None:
    bars = _bars()
    ticks = pd.DataFrame({"root": ["ES"], "tick_bp": [0.25]})
    full = build_context(bars, ticks)
    partial = build_context(bars.iloc[:140], ticks)
    assert_frame_equal(partial, full.iloc[:140], check_dtype=False)
