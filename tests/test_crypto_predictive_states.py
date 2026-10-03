"""Contracts for the new path representation, outcome times and geometric support."""

import numpy as np
import pandas as pd

from src.crypto.predictive_states import assignment, distribution, make_representation, outcome_labels


def test_middle_path_has_information_not_in_endpoints_and_is_prefix_causal() -> None:
    time = pd.date_range("2024-01-01", periods=50, freq="15min", tz="UTC")
    a = np.zeros(50)
    a[[8, 12, 16]] = [2, 4, 3]
    a[[28, 32, 36]] = [-3, -4, -2]
    frame = pd.DataFrame({"symbol": "BTCUSDT", "available_at": time,
                          "momentum_z_med_15m": a, "direction_alignment_15m": a})
    cfg = {"discovery_fit_end": "2024-01-01T06:00:00Z", "random_seed": 1,
           "representation_columns": {"trend": ["momentum_z_med_15m", "direction_alignment_15m"]},
           "path_columns": {"trend": ["momentum_z_med_15m", "direction_alignment_15m"]}}
    out = make_representation(frame, cfg)
    # Both observations have the same current value and the same 4h-old value.
    np.testing.assert_allclose(out["X1_endpoints"][20], out["X1_endpoints"][40])
    assert not np.allclose(out["X2_ordered_path"][20], out["X2_ordered_path"][40])
    before = {key: value[40].copy() for key, value in out.items() if key.startswith("X")}
    frame.loc[41:, "momentum_z_med_15m"] = 999
    later = make_representation(frame, cfg)
    for key, value in before.items():
        np.testing.assert_allclose(value, later[key][40])


def test_distribution_labels_and_assignment() -> None:
    outcome = outcome_labels(np.array([-.02, .02]), np.array([-.05, .05]),
                             np.array([.001, .00001]), np.array([.0006, .0006]),
                             (-1, -.25, .25, 1))
    assert outcome.tolist() == [[0, 0, 1], [4, 4, 0]]
    prior = distribution(outcome)
    assert all(np.isclose(p.sum(), 1) for p in prior)
    x = np.array([[0., 0.], [2., 2.]], dtype=np.float32)
    centers = np.array([[0., 0.], [2., 2.]], dtype=np.float32)
    ids, distances = assignment(x, centers)
    assert ids.tolist() == [0, 1]
    np.testing.assert_allclose(distances, 0, atol=1e-3)
