"""Research lifecycle checks for conflicting prototypes and fixed holding."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from crypto_full_pattern_stage import positions_from_candidates


def test_opposite_evidence_keeps_flat_and_fixed_hold_does_not_reenter() -> None:
    frame = pd.DataFrame({"symbol_code": [0] * 20})
    scores = np.zeros((20, 2), dtype=float)
    scores[0] = [.95, .95]  # equally strong opposite candidates
    scores[1, 0] = .95
    scores[2:, 1] = .95
    enter = np.array([.8, .8])
    exit_ = np.array([.7, .7])
    reliability = np.array([1.0, 1.0])
    directions = np.array([1, -1])
    positions, conflict = positions_from_candidates(scores, enter, exit_, reliability, directions, frame, hysteresis=False)
    assert conflict[0] and positions[0] == 0
    assert np.all(positions[1:17] == 1)
    assert positions[17] == -1


def test_hysteresis_exits_after_minimum_age_when_match_fades() -> None:
    frame = pd.DataFrame({"symbol_code": [0] * 8})
    scores = np.zeros((8, 1), dtype=float)
    scores[0, 0] = .95
    position, _ = positions_from_candidates(scores, np.array([.8]), np.array([.7]),
                                           np.array([1.0]), np.array([1]), frame, hysteresis=True)
    assert np.all(position[:4] == 1)
    assert np.all(position[4:] == 0)
