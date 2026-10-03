"""Causal and cross-frequency contracts for the crypto feature library."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.crypto.features import FEATURE_NAMES, GROUPS, build_features


def _bars(n: int, minutes: int) -> pd.DataFrame:
    t = np.arange(n, dtype=np.int64) * minutes * 60_000 + 1_672_531_200_000
    log_price = np.log(100.0) + np.cumsum(0.001 * np.sin(np.arange(n) / 13) + 0.0002)
    close = np.exp(log_price)
    open_ = np.r_[100.0, close[:-1]]
    return pd.DataFrame({
        "open_time": t, "close_time": t + minutes * 60_000 - 1,
        "open": open_, "high": np.maximum(open_, close) * 1.001,
        "low": np.minimum(open_, close) * 0.999, "close": close,
        "quote_volume": 100_000 + np.arange(n) % 17 * 1000,
        "trade_count": 100 + np.arange(n) % 11,
        "taker_buy_quote_volume": (100_000 + np.arange(n) % 17 * 1000) * 0.51,
    })


def test_feature_registry_and_prefix_causality() -> None:
    assert len(GROUPS) == 7
    assert all(len(group) == 8 for group in GROUPS.values())
    for frequency, minutes in (("5m", 5), ("15m", 15)):
        bars = _bars(1300, minutes)
        whole = build_features(bars, symbol="TEST", frequency=frequency)
        prefix = build_features(bars.iloc[:1000].copy(), symbol="TEST", frequency=frequency)
        assert whole["available_at"].iloc[0] == pd.Timestamp(bars.close_time.iloc[0] + 1, unit="ms", tz="UTC")
        np.testing.assert_allclose(
            whole[FEATURE_NAMES].iloc[:1000].to_numpy(),
            prefix[FEATURE_NAMES].to_numpy(),
            equal_nan=True,
        )
        assert whole[FEATURE_NAMES].iloc[-1].notna().sum() >= 50
