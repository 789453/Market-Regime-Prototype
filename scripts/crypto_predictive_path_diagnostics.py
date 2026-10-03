"""Predeclared X0/X1/X2 direct-model path comparison; no test-based reselection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.crypto_predictive_states_stage import (CFG, SCOPE, direct_fit, direct_predict,
                                                      load_panel, period_indices, score_row)
from scripts.crypto_predictive_states_diagnostics import paired_week
from src.crypto.predictive_states import log_loss, make_representation, outcome_labels


def main() -> None:
    panel = load_panel()
    periods = period_indices(panel)
    labels = outcome_labels(panel.return4.to_numpy(), panel.return24.to_numpy(),
                            panel.future_rv4.to_numpy(), panel.rv_slow_15m.to_numpy(),
                            tuple(CFG["return_bin_edges"]))
    rng = np.random.default_rng(CFG["random_seed"])
    for name in ("fit", "inner"):
        idx = periods[name]
        rng.choice(idx, size=max(1, len(idx) // CFG["outcome_sample_stride"]), replace=False)
    discovery = periods["discovery"]
    sampled = np.sort(rng.choice(discovery, size=max(1, len(discovery) // CFG["outcome_sample_stride"]), replace=False))
    reps = make_representation(panel, CFG)
    test = periods["test"]
    original = pd.read_parquet(SCOPE / "test_predictions.parquet",
                               columns=["symbol", "available_at", "direct_logloss",
                                        "direct_4h_return_logloss", "direct_24h_return_logloss", "direct_4h_vol_logloss"])
    original = original.rename(columns={"direct_logloss": "X1_overall", "direct_4h_return_logloss": "X1_return4",
                                        "direct_24h_return_logloss": "X1_return24", "direct_4h_vol_logloss": "X1_vol4"})
    results = []
    pairs = []
    importance_rows = []
    for name in ("X0_current", "X2_ordered_path"):
        model = direct_fit(reps[name], labels, sampled)
        features = []
        for group, current in CFG["representation_columns"].items():
            features.extend((group, "current", feature) for feature in current)
            if name == "X2_ordered_path" and group in CFG["path_columns"]:
                for lag in (16, 12, 8, 4):
                    features.extend((group, "middle_path" if lag < 16 else "4h_endpoint", f"{feature}__lag{lag}")
                                    for feature in CFG["path_columns"][group])
                for extreme in ("argmax", "argmin"):
                    features.extend((group, "event_order", f"{feature}__{extreme}")
                                    for feature in CFG["path_columns"][group])
        assert len(features) == reps[name].shape[1]
        for task, estimator in enumerate(model):
            gain = estimator.booster_.feature_importance(importance_type="gain")
            for (group, layer, feature), value in zip(features, gain):
                importance_rows.append({"model": name, "target": ("return4", "return24", "vol4")[task],
                                        "group": group, "layer": layer, "feature": feature, "gain": float(value)})
        p = direct_predict(model, reps[name][test])
        results.append(score_row(name + "_direct_exploratory", "test", labels[test], p, panel.available_at.iloc[test]))
        part = panel.iloc[test][["symbol", "available_at"]].copy()
        losses = log_loss(labels[test], p)
        part["overall"] = losses.mean(axis=1)
        part["return4"] = losses[:, 0]
        part["return24"] = losses[:, 1]
        part["vol4"] = losses[:, 2]
        part = part.merge(original, on=["symbol", "available_at"], validate="one_to_one")
        part["date"] = part.available_at.dt.floor("D")
        for metric, old in (("overall", "X1_overall"), ("return4", "X1_return4"),
                            ("return24", "X1_return24"), ("vol4", "X1_vol4")):
            value = paired_week(part, metric, old)
            pairs.append({"model_minus_X1": name, "target": metric, "daily_mean": value[0],
                          "week_low": value[1], "week_high": value[2]})
    pd.DataFrame(results).to_csv(SCOPE / "direct_path_test_scores_exploratory.csv", index=False)
    pd.DataFrame(pairs).to_csv(SCOPE / "direct_path_paired_exploratory.csv", index=False)
    importance = pd.DataFrame(importance_rows)
    importance["fraction_of_task_gain"] = importance.gain / importance.groupby(["model", "target"]).gain.transform("sum")
    importance.to_csv(SCOPE / "direct_path_feature_gain.csv", index=False)
    print(pd.DataFrame(results).to_string(index=False))
    print(pd.DataFrame(pairs).to_string(index=False))


if __name__ == "__main__":
    main()
