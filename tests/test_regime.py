import pandas as pd

from src.regime import detector_gate, gate_curve


def _context():
    return pd.DataFrame({
        "er_20": [0.1, 0.9], "vol_ratio": [0.7, 1.4], "sigma_pct": [0.3, 0.5],
        "liq_z": [0.0, 1.5], "session_phase": ["ASIA", "US_MORN"],
        "is_month_end": [False, True], "is_quarter_end": [False, False],
    })


def test_detector_specific_gates_are_bounded_and_signed():
    frame = _context()
    trend = detector_gate(frame, "D02")
    reversion = detector_gate(frame, "D01")
    assert trend.between(0.2, 1.5).all() and reversion.between(0.2, 1.5).all()
    assert trend.iloc[1] > trend.iloc[0]
    assert reversion.iloc[0] > reversion.iloc[1]


def test_gate_curve_detects_smooth_monotonic_relation():
    frame = pd.DataFrame({"detector_gate": range(100), "outcome": range(100)})
    _, metrics = gate_curve(frame)
    assert metrics["isotonic_r2"] > 0.99
    assert metrics["adjacent_jump_ratio"] < 0.2
    assert metrics["passed"]
