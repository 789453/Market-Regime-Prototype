import pandas as pd

from src.structure import online_zigzag


def test_zigzag_never_backdates_confirmation():
    close = [100, 101, 102, 103, 102, 101, 100, 101, 102]
    frame = pd.DataFrame({
        "root": "ES", "timestamp": pd.date_range("2026-01-01", periods=len(close), freq="5min", tz="UTC"),
        "close": close, "volume": 100.0, "is_valid": True, "sigma_hat": 0.002,
    })
    pivots = online_zigzag(frame, kappa=1.5, n_ref=12)
    assert not pivots.empty
    assert (pivots["confirm_time"] >= pivots["event_time"]).all()
