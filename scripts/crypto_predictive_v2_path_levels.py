"""Small-capacity ablation of own current, 4h endpoint and 1h recent path."""

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
from src.crypto.predictive_states_v2 import compact_features

OUT = ROOT / "reports/crypto/predictive_states_v2"


def main() -> None:
    panel = load_panel()
    period = period_indices(panel)
    y = outcome_labels(panel.return4.to_numpy(), panel.return24.to_numpy(), panel.future_rv4.to_numpy(),
                       panel.rv_slow_15m.to_numpy(), tuple(CFG["return_bin_edges"]))
    symbol = panel.symbol_code.to_numpy(dtype=np.int8)
    compact, names, split = compact_features(make_representation(panel, CFG)["X2_ordered_path"],
                                             symbol, panel.rv_slow_15m.to_numpy())
    rng = np.random.default_rng(CFG["random_seed"])
    train = np.sort(rng.choice(period["discovery"], len(period["discovery"]) // 4, replace=False))
    val, test = period["validation"], period["test"]
    index = np.concatenate((train, val, test))
    asset = np.eye(12, dtype=np.float32)[symbol[index]]
    base = np.r_[split["market"], split["clock"], split["rv"]]
    current = np.arange(0, 18, 3)
    endpoint = np.arange(1, 18, 3)
    recent = np.arange(2, 18, 3)
    variants = {
        "current": np.r_[current, base],
        "current_endpoint": np.r_[current, endpoint, base],
        "current_endpoint_recent": np.r_[current, endpoint, recent, base],
    }
    ntrain, nval = len(train), len(val)
    reference = joblib.load(OUT / "conditional_models.joblib")["market_clock_background"]["models"]
    ref_x = np.column_stack((compact[index][:, base], asset))
    ref_val = log_loss(y[val], predict(reference, ref_x[ntrain:ntrain + nval])).mean(axis=1)
    ref_test = log_loss(y[test], predict(reference, ref_x[ntrain + nval:])).mean(axis=1)
    rows, differences = [], []
    for name, cols in variants.items():
        x = np.column_stack((compact[index][:, cols], asset)).astype(np.float32)
        models = fit_models(x, y[index], np.arange(ntrain), CFG["random_seed"])
        joblib.dump({"models": models, "feature_names": [names[j] for j in cols] +
                     [f"symbol_{s}" for s in range(12)]}, OUT / f"path_level_{name}.joblib", compress=3)
        for phase, local_index, global_index, ref in (
            ("validation", np.arange(ntrain, ntrain + nval), val, ref_val),
            ("test_exploratory", np.arange(ntrain + nval, len(index)), test, ref_test)):
            p = predict(models, x[local_index])
            rows.append({"model": name, "phase": phase,
                         **score_row(name, phase, y[global_index], p, panel.available_at.iloc[global_index])})
            loss = log_loss(y[global_index], p).mean(axis=1)
            mean, low, high = paired_block_interval(loss, ref, panel.available_at.iloc[global_index])
            differences.append({"model": name, "phase": phase, "minus_market_clock_rv": mean,
                                "week_low": low, "week_high": high})
        print(name, [(r["phase"], round(r["logloss"], 6)) for r in rows[-2:]], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "path_level_scores.csv", index=False)
    pd.DataFrame(differences).to_csv(OUT / "path_level_paired.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    for phase in ("validation", "test_exploratory"):
        subset = pd.DataFrame(rows).loc[lambda d: d.phase == phase]
        ax.plot(subset.model, subset.logloss, marker="o", label=phase)
    ax.set_ylabel("Three-task log loss")
    ax.set_title("Own-path increment beyond known market, clock and risk")
    ax.tick_params(axis="x", rotation=15)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "06_path_level_ablation.png", dpi=160)


if __name__ == "__main__":
    main()
