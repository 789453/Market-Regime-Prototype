import numpy as np
import pandas as pd

from src.clean import add_valid_mask, resample_5m


def _minute_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2026-03-09 13:30:00Z", periods=6, freq="min")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0, 100.0, 101.0, 102.0, 102.0, 103.0],
            "high": [100.0, 100.0, 102.0, 103.0, 102.0, 104.0],
            "low": [100.0, 100.0, 100.5, 101.5, 102.0, 102.5],
            "close": [100.0, 100.0, 101.5, 102.5, 102.0, 103.5],
            "volume": [0.0, 0.0, 10.0, 20.0, 0.0, 30.0],
            "bar_count": [0, 0, 2, 4, 0, 6],
            "wap": [100.0, 100.0, 101.0, 102.0, 102.0, 103.0],
            "contract_con_id": [1] * 6,
            "contract_expiry": [20260320] * 6,
            "local_symbol": ["ESH6"] * 6,
            "root": ["ES"] * 6,
        }
    )


def test_valid_mask_matches_document_definition() -> None:
    result = add_valid_mask(_minute_frame())
    assert result["is_filled"].tolist() == [True, True, False, False, True, False]
    assert result["is_real_1m"].sum() == 3


def test_resample_uses_only_valid_minutes_for_ohlc_and_wap() -> None:
    result = resample_5m(_minute_frame())
    first = result.iloc[0]

    assert first["open"] == 101.0
    assert first["high"] == 103.0
    assert first["low"] == 100.5
    assert first["close"] == 102.5
    assert first["volume"] == 30.0
    assert first["bar_count"] == 6
    assert first["n_valid_1m"] == 2
    assert bool(first["is_valid"])
    assert np.isclose(first["wap"], (101.0 * 10.0 + 102.0 * 20.0) / 30.0)

