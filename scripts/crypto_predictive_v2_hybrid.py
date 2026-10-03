"""Restricted combination: soft historical identity as optional conditional input."""

from __future__ import annotations

import json
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
TASKS = ("4h_return", "24h_return", "4h_vol")


def main() -> None:
    panel = load_panel()
    period = period_indices(panel)
    y = outcome_labels(panel.return4.to_numpy(), panel.return24.to_numpy(), panel.future_rv4.to_numpy(),
                       panel.rv_slow_15m.to_numpy(), tuple(CFG["return_bin_edges"]))
    symbol = panel.symbol_code.to_numpy(dtype=np.int8)
    rep = make_representation(panel, CFG)
    compact, names, _ = compact_features(rep["X2_ordered_path"], symbol, panel.rv_slow_15m.to_numpy())
    rng = np.random.default_rng(CFG["random_seed"])
    train = np.sort(rng.choice(period["discovery"], len(period["discovery"]) // 4, replace=False))
    val, test = period["validation"], period["test"]
    index = np.concatenate((train, val, test))
    asset = np.eye(12, dtype=np.float32)[symbol[index]]
    base = np.column_stack((compact[index], asset)).astype(np.float32)
    resolution = joblib.load(OUT / "resolution_chosen_model.joblib")
    hard, distance, weight, margin = center_membership(rep["X1_endpoints"][index, :-4],
                                                        resolution["centers"], resolution["temperature"])
    geometry = np.column_stack((distance, margin)).astype(np.float32)
    variants = {
        "compact_with_geometry_margin": (np.column_stack((base, geometry)),
                                         names + [f"symbol_{s}" for s in range(12)] + ["prototype_distance", "prototype_margin"]),
        "compact_with_soft_regions": (np.column_stack((base, weight)),
                                      names + [f"symbol_{s}" for s in range(12)] +
                                      [f"prototype_soft_{j}" for j in range(weight.shape[1])]),
    }
    ntrain, nval = len(train), len(val)
    train_local = np.arange(ntrain)
    val_local = np.arange(ntrain, ntrain + nval)
    test_local = np.arange(ntrain + nval, len(index))
    old_models = joblib.load(OUT / "conditional_models.joblib")["compact_no_prototype"]["models"]
    base_val = predict(old_models, base[val_local])
    base_test = predict(old_models, base[test_local])
    base_val_loss = log_loss(y[val], base_val).mean(axis=1)
    base_test_loss = log_loss(y[test], base_test).mean(axis=1)
    scores, comparisons = [], []
    for model_name, (x, feature_names) in variants.items():
        model = fit_models(x, y[index], train_local, CFG["random_seed"])
        joblib.dump({"models": model, "feature_names": feature_names},
                    OUT / f"hybrid_{model_name}.joblib", compress=3)
        gains = pd.DataFrame({"feature": feature_names,
                              **{TASKS[j]: model[j].booster_.feature_importance(importance_type="gain")
                                 for j in range(3)}})
        gains.to_csv(OUT / f"hybrid_gain_{model_name}.csv", index=False)
        for phase, local_idx, global_idx, base_loss in (
            ("validation", val_local, val, base_val_loss),
            ("test_exploratory", test_local, test, base_test_loss)):
            p = predict(model, x[local_idx])
            scores.append({"model": model_name, "phase": phase,
                           **score_row(model_name, phase, y[global_idx], p, panel.available_at.iloc[global_idx])})
            loss = log_loss(y[global_idx], p).mean(axis=1)
            mean, low, high = paired_block_interval(loss, base_loss, panel.available_at.iloc[global_idx])
            comparisons.append({"model": model_name, "phase": phase,
                                "minus_compact_no_prototype": mean, "week_low": low, "week_high": high})
            if phase == "test_exploratory":
                output = panel.iloc[test][["symbol", "available_at"]].copy()
                output["prototype_nearest"] = hard[test_local]
                output["distance"] = distance[test_local]
                output["margin"] = margin[test_local]
                output["loss"] = loss
                output.to_parquet(OUT / f"hybrid_{model_name}_test_rows.parquet", index=False, compression="zstd")
        print(model_name, [(s["phase"], round(s["logloss"], 6)) for s in scores[-2:]], flush=True)
    pd.DataFrame(scores).to_csv(OUT / "hybrid_scores.csv", index=False)
    pd.DataFrame(comparisons).to_csv(OUT / "hybrid_paired.csv", index=False)
    (OUT / "hybrid_manifest.json").write_text(json.dumps({"resolution_k": int(len(resolution["centers"])),
        "geometry": "X1 clock-free; input-only soft membership", "conditional_capacity": "75 trees, 7 leaves",
        "test_status": "exploratory reuse"}, indent=2), encoding="utf-8")
    plot = pd.DataFrame(comparisons)
    fig, ax = plt.subplots(figsize=(8, 4))
    for phase in ("validation", "test_exploratory"):
        view = plot.loc[plot.phase == phase]
        ax.errorbar(view.model, view.minus_compact_no_prototype,
                    yerr=[view.minus_compact_no_prototype - view.week_low,
                          view.week_high - view.minus_compact_no_prototype],
                    marker="o", capsize=3, linestyle="none", label=phase)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Loss minus compact model; lower is better")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
    ax.set_title("Does geometric identity add conditional information?")
    fig.tight_layout()
    fig.savefig(OUT / "05_geometry_identity_increment.png", dpi=160)


if __name__ == "__main__":
    main()
