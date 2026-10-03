"""Exploratory no-clock direct-model ablation; cannot promote a new test winner."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.crypto_predictive_states_stage import (CFG, SCOPE, direct_fit, direct_predict,
                                                      load_panel, period_indices, score_row)
from src.crypto.predictive_states import make_representation, outcome_labels


def main() -> None:
    panel = load_panel()
    periods = period_indices(panel)
    y = outcome_labels(panel.return4.to_numpy(), panel.return24.to_numpy(),
                       panel.future_rv4.to_numpy(), panel.rv_slow_15m.to_numpy(),
                       tuple(CFG["return_bin_edges"]))
    rng = np.random.default_rng(CFG["random_seed"])
    for name in ("fit", "inner", "discovery"):
        idx = periods[name]
        sample = np.sort(rng.choice(idx, size=max(1, len(idx) // CFG["outcome_sample_stride"]), replace=False))
        if name == "discovery":
            train = sample
    reps = make_representation(panel, CFG)
    rows = []
    for name in ("X0_current", "X1_endpoints", "X2_ordered_path"):
        # time_context is the final four columns in every registered representation.
        x = reps[name][:, :-4]
        models = direct_fit(x, y, train)
        for period in ("validation", "test"):
            idx = periods[period]
            p = direct_predict(models, x[idx])
            rows.append(score_row(name + "_direct_no_clock", period, y[idx], p, panel.available_at.iloc[idx]))
    result = pd.DataFrame(rows)
    result.to_csv(SCOPE / "direct_no_clock_scores_exploratory.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
