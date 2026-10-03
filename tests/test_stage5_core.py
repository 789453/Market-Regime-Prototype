import numpy as np
import pandas as pd

from src.backtest import run_event_backtest
from src.fusion import effective_detector_count, fuse_signals
from src.metrics import performance_summary
from src.sizing import size_instruments


def _index(n=6):
    return pd.MultiIndex.from_product(
        [["ES"], pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")],
        names=["root", "timestamp"],
    )


def _signal(direction, magnitude=1.0, confidence=1.0, horizon=2):
    idx = _index(len(direction))
    return pd.DataFrame(
        {
            "direction": direction,
            "magnitude": np.where(np.asarray(direction) != 0, magnitude, 0.0),
            "horizon": horizon,
            "confidence": np.where(np.asarray(direction) != 0, confidence, 0.0),
            "gate_score": 1.0,
        },
        index=idx,
    )


def test_fusion_formula_and_causal_horizon_decay():
    a = _signal([1, 0, 0, 0], horizon=2)
    out, _ = fuse_signals({"D01": a}, tau=0.5, persist_horizon=True)
    # Trigger: 1 / (1 + .5); next bar has half the vote and confidence.
    assert np.isclose(out.iloc[0].mu, 2 / 3)
    assert np.isclose(out.iloc[1].mu, 0.5 / 1.0)
    assert out.iloc[2].mu == 0


def test_duplicate_cluster_is_averaged_before_fusion():
    a = _signal([1, 0, 0, 0])
    out1, _ = fuse_signals({"D01": a}, tau=0.5)
    out2, dep = fuse_signals({"D01": a, "D02": a.copy()}, tau=0.5, causal_clusters=False)
    assert np.allclose(out1.mu, out2.mu)
    assert dep.loc["D01", "cluster"] == dep.loc["D02", "cluster"]


def test_effective_detector_count_bounds():
    assert np.isclose(effective_detector_count(np.eye(3)), 3.0)
    assert np.isclose(effective_detector_count(np.ones((3, 3))), 1.0)


def test_sizing_is_bounded_and_breaker_only_shrinks():
    idx = _index(4)
    fused = pd.DataFrame({"mu": [10.0] * 4, "active_horizon": [24.0] * 4}, index=idx)
    context = pd.DataFrame(
        {
            "sigma_hat": [0.001] * 4,
            "sigma_pct": [0.5, 0.5, 1.0, 0.5],
            "spread_bp": [1.0] * 4,
        },
        index=idx,
    )
    out = size_instruments(fused, context, max_weight=2.0)
    assert out.target.abs().max() <= 2.0
    assert np.isclose(out.iloc[2].target / out.iloc[1].target, 0.3)


def test_event_loop_executes_on_next_bar_and_splits_execution_return():
    idx = _index(4)
    bars = pd.DataFrame(
        {
            "open": [100, 100, 110, 110],
            "close": [100, 110, 110, 110],
            "wap": [100, 105, 110, 110],
            "volume": 1000.0,
            "is_valid": True,
            "bar_in_session": [3, 4, 5, 6],
            "session_id": "s1",
            "session_phase": "US_CASH",
            "is_roll_boundary": False,
            "ret": [np.nan, np.log(1.1), 0.0, 0.0],
            "overnight_ret": np.nan,
        },
        index=idx,
    )
    decisions = pd.DataFrame(
        {"target": [1.0, 1.0, 1.0, 1.0], "band": 0.0, "round_trip_cost_bp": 0.0},
        index=idx,
    )
    out = run_event_backtest(bars, decisions, delay_bars=1, avoid_session_edge_bars=0)
    assert out.iloc[0].position == 0
    assert out.iloc[1].position == 1
    # New position is entered at wap=105, so it earns only wap-to-close.
    assert np.isclose(out.iloc[1].gross_return, np.log(110 / 105))


def test_multi_bar_delay_does_not_drop_pending_decisions():
    idx = _index(5)
    bars = pd.DataFrame(
        {
            "open": 100.0, "close": 100.0, "wap": 100.0, "volume": 1000.0,
            "is_valid": True, "bar_in_session": np.arange(3, 8), "session_id": "s1",
            "session_phase": "US_CASH", "is_roll_boundary": False,
            "ret": [np.nan, 0.0, 0.0, 0.0, 0.0], "overnight_ret": np.nan,
        }, index=idx,
    )
    decisions = pd.DataFrame(
        {"target": 1.0, "band": 0.0, "signal_active": True, "round_trip_cost_bp": 0.0},
        index=idx,
    )
    out = run_event_backtest(bars, decisions, delay_bars=3, avoid_session_edge_bars=0)
    assert out.iloc[:3].position.eq(0).all()
    assert out.iloc[3].position == 1
    assert out.iloc[3].execution_delay_bars == 3


def test_metrics_are_finite_for_nonconstant_daily_series():
    r = pd.Series([0.01, -0.005, 0.008, -0.002, 0.004])
    m = performance_summary(r)
    assert np.isfinite(m["annualized_volatility"])
    assert "sharpe_ci_low" in m and "max_drawdown" in m
