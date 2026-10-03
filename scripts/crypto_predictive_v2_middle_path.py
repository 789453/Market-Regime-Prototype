"""Explicit intermediate 2h/3h nodes and local acceleration as compact predictors."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.crypto_predictive_states_stage import CFG, load_panel, paired_block_interval, period_indices, score_row
from scripts.crypto_predictive_v2_conditional import fit_models, predict
from src.crypto.predictive_states import log_loss, make_representation, outcome_labels
from src.crypto.predictive_states_v2 import center_membership, compact_features

OUT = ROOT / "reports/crypto/predictive_states_v2"
GROUPS = ("vol_strength", "vol_asymmetry", "path_geometry", "trend", "efficiency", "participation")
TASKS = ("4h_return", "24h_return", "4h_vol")


def middle_summaries(x2: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Each column is a difference of already causal, normalized path nodes."""
    fields = []
    names = []
    for g, group in enumerate(GROUPS):
        block = x2[:, 16 * g:16 * (g + 1)]
        now = block[:, 2:4]
        three_h = block[:, 6:8]
        two_h = block[:, 8:10]
        one_h = block[:, 10:12]
        fields += [(now - three_h).mean(axis=1), (now - two_h).mean(axis=1),
                   ((now - one_h) - (one_h - two_h)).mean(axis=1)]
        names += [f"{group}_change3h", f"{group}_change2h", f"{group}_recent_acceleration"]
    return np.stack(fields, axis=1).astype(np.float32), names


def main() -> None:
    panel = load_panel()
    period = period_indices(panel)
    y = outcome_labels(panel.return4.to_numpy(), panel.return24.to_numpy(), panel.future_rv4.to_numpy(),
                       panel.rv_slow_15m.to_numpy(), tuple(CFG["return_bin_edges"]))
    symbol = panel.symbol_code.to_numpy(dtype=np.int8)
    rep = make_representation(panel, CFG)
    compact, names, _ = compact_features(rep["X2_ordered_path"], symbol,
                                         panel.rv_slow_15m.to_numpy())
    intermediate, middle_names = middle_summaries(rep["X2_ordered_path"])
    rng = np.random.default_rng(CFG["random_seed"])
    train = np.sort(rng.choice(period["discovery"], len(period["discovery"]) // 4, replace=False))
    val, test = period["validation"], period["test"]
    index = np.concatenate((train, val, test))
    asset = np.eye(12, dtype=np.float32)[symbol[index]]
    base = np.column_stack((compact[index], asset)).astype(np.float32)
    midpoint = intermediate[index][:, np.array([0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16])].astype(np.float32)
    acceleration = intermediate[index][:, np.arange(2, 18, 3)]
    qmodel = joblib.load(OUT / "resolution_chosen_model.joblib")
    _, distance, weights, margin = center_membership(rep["X1_endpoints"][index, :-4],
                                                      qmodel["centers"], qmodel["temperature"])
    variants = {
        "compact_middle12": np.column_stack((base, midpoint)),
        "compact_middle12_accel6": np.column_stack((base, midpoint, acceleration)),
        "compact_middle12_distance": np.column_stack((base, midpoint, distance, margin)),
    }
    ntrain, nval = len(train), len(val)
    old_model = joblib.load(OUT / "conditional_models.joblib")["compact_no_prototype"]["models"]
    old_val = log_loss(y[val], predict(old_model, base[ntrain:ntrain + nval])).mean(axis=1)
    old_test = log_loss(y[test], predict(old_model, base[ntrain + nval:])).mean(axis=1)
    rows, paired = [], []
    for name, x in variants.items():
        models = fit_models(x, y[index], np.arange(ntrain), CFG["random_seed"])
        feature_names = names + [f"symbol_{s}" for s in range(12)]
        feature_names += [n for n in middle_names if not n.endswith("recent_acceleration")]
        if "accel6" in name:
            feature_names += [n for n in middle_names if n.endswith("recent_acceleration")]
        if "distance" in name:
            feature_names += ["prototype_distance", "prototype_margin"]
        joblib.dump({"models": models, "feature_names": feature_names},
                    OUT / f"middle_{name}.joblib", compress=3)
        gain = pd.DataFrame({"feature": feature_names,
                             **{TASKS[j]: models[j].booster_.feature_importance(importance_type="gain")
                                for j in range(3)}})
        gain.to_csv(OUT / f"middle_gain_{name}.csv", index=False)
        for phase, local, global_index, old_loss in (
            ("validation", np.arange(ntrain, ntrain + nval), val, old_val),
            ("test_exploratory", np.arange(ntrain + nval, len(index)), test, old_test)):
            p = predict(models, x[local])
            rows.append({"model": name, "phase": phase,
                         **score_row(name, phase, y[global_index], p, panel.available_at.iloc[global_index])})
            loss = log_loss(y[global_index], p)
            mean, low, high = paired_block_interval(loss.mean(axis=1), old_loss,
                                                    panel.available_at.iloc[global_index])
            paired.append({"model": name, "phase": phase, "minus_compact_no_prototype": mean,
                           "week_low": low, "week_high": high})
            if phase == "test_exploratory":
                output = panel.iloc[test][["symbol", "available_at"]].copy()
                for task, target in enumerate(TASKS):
                    output[f"{target}_loss"] = loss[:, task]
                output.to_parquet(OUT / f"middle_{name}_test_rows.parquet", index=False, compression="zstd")
        print(name, [(r["phase"], round(r["logloss"], 6)) for r in rows[-2:]], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "middle_path_scores.csv", index=False)
    pd.DataFrame(paired).to_csv(OUT / "middle_path_paired.csv", index=False)
    registry = []
    for group in GROUPS:
        for field, formula, duplicate in (
            (f"{group}_change3h", "mean(current_2 - lag_3h_2)", f"{group} current and 4h/2h changes"),
            (f"{group}_change2h", "mean(current_2 - lag_2h_2)", f"{group} current and 4h/3h changes"),
            (f"{group}_recent_acceleration", "mean[(current_2-lag_1h_2)-(lag_1h_2-lag_2h_2)]", f"{group} 1h/2h changes")):
            registry.append({"field": field, "group": group, "formula": formula,
                             "window": "completed 15m snapshots over past 4h", "unit": "group-scaled midrank difference",
                             "expected_direction": "unknown; no return sign prespecified",
                             "possible_duplicate": duplicate,
                             "missing": "base feature mapped by training midrank; first 16 snapshots excluded",
                             "available_at": "current 15m bar completion"})
    pd.DataFrame(registry).to_csv(OUT / "middle_path_feature_registry.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    for phase in ("validation", "test_exploratory"):
        view = pd.DataFrame(rows).loc[lambda d: d.phase == phase]
        ax.plot(view.model, view.logloss, marker="o", label=phase)
    ax.tick_params(axis="x", rotation=15)
    ax.set_ylabel("Three-target distribution log loss")
    ax.set_title("Explicit middle-path summaries: controlled additions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "09_middle_path_ablation.png", dpi=160)


if __name__ == "__main__":
    main()
