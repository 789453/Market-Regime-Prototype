import pandas as pd

from src.returns import build_labels


def test_label_invalid_across_roll_boundary() -> None:
    frame = pd.DataFrame(
        {
            "root": ["ES"] * 8,
            "timestamp": pd.date_range("2026-03-02 23:00Z", periods=8, freq="5min"),
            "close": range(100, 108),
            "is_valid": [True] * 8,
            "is_roll_boundary": [False, False, False, True, False, False, False, False],
            "session_id": [pd.Timestamp("2026-03-03").date()] * 8,
            "sigma_hat": [0.01] * 8,
        }
    )
    labels = build_labels(frame, horizons=(3,))
    assert not bool(labels.loc[0, "label_valid_3"])
    assert not bool(labels.loc[1, "label_valid_3"])
    assert not bool(labels.loc[2, "label_valid_3"])
    assert bool(labels.loc[3, "label_valid_3"])

