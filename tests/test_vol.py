import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from src.vol import (
    bipower_variation,
    expanding_har_forecast,
    estimate_causal_seasonality,
    finalize_sigma_hat,
)


def test_bipower_variation_matches_formula() -> None:
    returns = np.array([0.01, -0.02, 0.03, -0.01])
    expected = np.pi / 2 * 4 / 3 * np.sum(np.abs(returns[1:]) * np.abs(returns[:-1]))
    assert np.isclose(bipower_variation(returns), expected)


def _daily_panel(periods: int = 65) -> pd.DataFrame:
    sessions = pd.bdate_range("2026-01-01", periods=periods)
    rows = []
    for root, scale in [("ES", 1.0), ("NQ", 1.4), ("RTY", 1.2)]:
        variance = 1e-5 * scale * np.exp(0.25 * np.sin(np.arange(periods) / 6))
        for date, value in zip(sessions, variance):
            rows.append(
                {
                    "root": root,
                    "session_id": date.date(),
                    "continuous_var": value,
                    "jump_var": value * 0.05,
                    "session_ret": -0.01 if date.day % 9 == 0 else 0.002,
                    "n_core_returns": 100,
                }
            )
    return pd.DataFrame(rows)


def test_expanding_har_is_prefix_invariant() -> None:
    daily = _daily_panel()
    full = expanding_har_forecast(daily, min_sessions=25)
    cutoff = sorted(daily["session_id"].unique())[50]
    partial = expanding_har_forecast(daily.loc[daily["session_id"].le(cutoff)], min_sessions=25)
    expected = full.loc[full["session_id"].le(cutoff), ["root", "session_id", "har_log_var"]].reset_index(drop=True)
    actual = partial[["root", "session_id", "har_log_var"]].reset_index(drop=True)
    assert_series_equal(actual["har_log_var"], expected["har_log_var"], check_names=False)


def _seasonal_bars(sessions: int = 8) -> pd.DataFrame:
    rows = []
    for day, session in enumerate(pd.bdate_range("2026-01-01", periods=sessions)):
        for slot, multiplier in enumerate([0.7, 1.0, 1.4, 1.0]):
            rows.append(
                {
                    "root": "ES",
                    "session_id": session.date(),
                    "local_minute": 18 * 60 + slot * 5,
                    "ret": (1 if (day + slot) % 2 else -1) * 0.001 * multiplier,
                    "sigma_day": 0.001,
                }
            )
    return pd.DataFrame(rows)


def test_seasonality_and_sigma_are_prefix_invariant() -> None:
    bars = _seasonal_bars()
    full_seas = estimate_causal_seasonality(bars, min_sessions=3, update_every=1, smooth_width=3)
    cutoff = 6 * 4
    partial = bars.iloc[:cutoff].copy()
    partial_seas = estimate_causal_seasonality(partial, min_sessions=3, update_every=1, smooth_width=3)
    assert_series_equal(partial_seas, full_seas.iloc[:cutoff], check_names=False)

    full_vol = finalize_sigma_hat(bars.assign(seas=full_seas), ewma_lambda=0.8, theta=0.4)
    partial_vol = finalize_sigma_hat(partial.assign(seas=partial_seas), ewma_lambda=0.8, theta=0.4)
    assert_series_equal(
        partial_vol["sigma_hat"], full_vol["sigma_hat"].iloc[:cutoff], check_names=False
    )

