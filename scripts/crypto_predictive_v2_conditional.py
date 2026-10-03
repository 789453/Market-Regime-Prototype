"""V2 route B: retain historical prototype identity, relax constant leaf probabilities."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.crypto_predictive_states_stage import (CFG, load_panel, paired_block_interval,
                                                    period_indices, score_row)
from src.crypto.predictive_states import log_loss, make_representation, outcome_labels, predict_states
from src.crypto.predictive_states_v2 import compact_features

OUT = ROOT / "reports/crypto/predictive_states_v2"
OLD = ROOT / "reports/crypto/predictive_states_v1"
TASKS = ("4h_return", "24h_return", "4h_vol")


def fit_models(x: np.ndarray, labels: np.ndarray, index: np.ndarray, seed: int) -> list[lgb.LGBMClassifier]:
    models = []
    for task in range(3):
        model = lgb.LGBMClassifier(n_estimators=75, learning_rate=.045, num_leaves=7,
                                   max_depth=4, min_child_samples=1500, reg_lambda=30,
                                   colsample_bytree=.9, subsample=.8, subsample_freq=1,
                                   n_jobs=6, verbosity=-1, random_state=seed + task)
        model.fit(x[index], labels[index, task])
        models.append(model)
    return models


def predict(models: list[lgb.LGBMClassifier], x: np.ndarray) -> list[np.ndarray]:
    return [m.predict_proba(x).astype(np.float32) for m in models]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    periods = period_indices(panel)
    y = outcome_labels(panel.return4.to_numpy(), panel.return24.to_numpy(), panel.future_rv4.to_numpy(),
                       panel.rv_slow_15m.to_numpy(), tuple(CFG["return_bin_edges"]))
    symbols = panel.symbol_code.to_numpy(dtype=np.int8)
    rep = make_representation(panel, CFG)
    compact, names, split = compact_features(rep["X2_ordered_path"], symbols,
                                             panel.rv_slow_15m.to_numpy())
    asset = np.eye(len(np.unique(symbols)), dtype=np.float32)[symbols]
    asset_names = [f"symbol_{code}" for code in range(asset.shape[1])]
    saved = joblib.load(OLD / "selected_models.joblib")
    state_model = saved["model"]
    print("loaded", len(panel), "compact features", len(names), flush=True)
    rng = np.random.default_rng(CFG["random_seed"])
    train = np.sort(rng.choice(periods["discovery"],
                               len(periods["discovery"]) // CFG["outcome_sample_stride"], replace=False))
    val, test = periods["validation"], periods["test"]
    all_index = np.concatenate((train, val, test))
    _, state, distance, support = predict_states(state_model, rep["X1_endpoints"][all_index],
                                                 symbols[all_index], split=True)
    leaf_ids = sorted({int(r["leaf"]) for r in state_model.medoids})
    state_onehot = np.eye(len(leaf_ids), dtype=np.float32)[np.searchsorted(leaf_ids, state)]
    state_names = [f"prototype_{leaf}" for leaf in leaf_ids]
    features = compact[all_index]
    own = split["own"]
    market = split["market"]
    clock = split["clock"]
    rv = split["rv"]
    base_cols = np.r_[market, clock, rv]
    compact_cols = np.r_[own, market, clock, rv]
    input_base = np.column_stack((features[:, base_cols], asset[all_index]))
    input_compact = np.column_stack((features[:, compact_cols], asset[all_index]))
    input_state = np.column_stack((input_compact, state_onehot))
    variants = {
        "market_clock_background": (input_base, [names[j] for j in base_cols] + asset_names),
        "compact_no_prototype": (input_compact, [names[j] for j in compact_cols] + asset_names),
        "compact_with_prototype": (input_state, [names[j] for j in compact_cols] + asset_names + state_names),
    }
    ntrain, nval = len(train), len(val)
    loc_train = np.arange(ntrain)
    loc_val = np.arange(ntrain, ntrain + nval)
    loc_test = np.arange(ntrain + nval, len(all_index))
    rows = []
    daily = []
    models_out = {}
    test_predictions = {}
    for variant, (x, feature_names) in variants.items():
        models = fit_models(x, y[all_index], loc_train, CFG["random_seed"])
        models_out[variant] = {"models": models, "feature_names": feature_names}
        for phase, local_idx, global_idx in (("validation", loc_val, val), ("test_exploratory", loc_test, test)):
            p = predict(models, x[local_idx])
            score = score_row(variant, phase, y[global_idx], p, panel.available_at.iloc[global_idx])
            rows.append({"model": variant, "phase": phase, **score})
            loss = log_loss(y[global_idx], p)
            dates = panel.available_at.iloc[global_idx].dt.floor("D").to_numpy()
            day = pd.DataFrame({"date": dates, "overall": loss.mean(axis=1),
                                **{name: loss[:, j] for j, name in enumerate(TASKS)}}).groupby("date").mean().reset_index()
            day["model"] = variant
            day["phase"] = phase
            daily.append(day)
            if phase == "test_exploratory":
                test_predictions[variant] = p
        gain = pd.DataFrame({"feature": feature_names,
                             **{TASKS[j]: models[j].booster_.feature_importance(importance_type="gain")
                                for j in range(3)}})
        gain.to_csv(OUT / f"conditional_gain_{variant}.csv", index=False)
        print(variant, [(r["phase"], round(r["logloss"], 6)) for r in rows[-2:]], flush=True)

    old = pd.read_csv(OLD / "all_scores.csv")
    for phase, old_period, index in (("validation", "validation", val),
                                      ("test_exploratory", "test", test)):
        for model, old_name in (("v1_X1_center", "X1_endpoints_split"),
                                 ("v1_X1_direct", "X1_endpoints_direct")):
            record = old.loc[(old.period == old_period) & (old.model == old_name)].iloc[0].to_dict()
            rows.append({**record, "phase": phase, "model": model})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "conditional_all_scores.csv", index=False)
    pd.concat(daily, ignore_index=True).to_csv(OUT / "conditional_daily_scores.csv", index=False)
    joblib.dump(models_out, OUT / "conditional_models.joblib", compress=3)

    # The same 2026 rows support paired, exploratory comparisons; these are not a fresh holdout.
    old_test = pd.read_parquet(OLD / "test_predictions.parquet",
                               columns=["symbol", "available_at", "state_logloss", "direct_logloss"])
    if not np.array_equal(old_test.available_at.to_numpy(), panel.available_at.iloc[test].to_numpy()):
        raise AssertionError("v1 test rows do not align")
    primary = test_predictions["compact_with_prototype"]
    lower = test_predictions["compact_no_prototype"]
    bg = test_predictions["market_clock_background"]
    lp = log_loss(y[test], primary).mean(axis=1)
    lower_loss = log_loss(y[test], lower).mean(axis=1)
    bg_loss = log_loss(y[test], bg).mean(axis=1)
    dates = panel.available_at.iloc[test]
    comparisons = {}
    for name, alternate in (("minus_compact_no_prototype", lower_loss),
                             ("minus_market_clock_background", bg_loss),
                             ("minus_v1_X1_center", old_test.state_logloss.to_numpy()),
                             ("minus_v1_X1_direct", old_test.direct_logloss.to_numpy())):
        comparisons[name] = paired_block_interval(lp, alternate, dates)
    (OUT / "conditional_paired.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    test_frame = panel.iloc[test][["symbol", "available_at"]].copy()
    test_frame["state"] = state[loc_test]
    test_frame["geometry_distance"] = distance[loc_test]
    test_frame["support90"] = support[loc_test]
    for model, p in test_predictions.items():
        loss = log_loss(y[test], p)
        for j, task in enumerate(TASKS):
            test_frame[f"{model}_{task}_loss"] = loss[:, j]
            for category in range(p[j].shape[1]):
                test_frame[f"{model}_{task}_p{category}"] = p[j][:, category]
    test_frame.to_parquet(OUT / "conditional_test_predictions.parquet", index=False, compression="zstd")
    gain = pd.read_csv(OUT / "conditional_gain_compact_with_prototype.csv")
    gain["family"] = np.where(gain.feature.str.startswith("prototype_"), "prototype",
                              np.where(gain.feature.str.startswith("market_"), "market",
                                       np.where(gain.feature.str.startswith("clock_"), "clock", "other")))
    gain.groupby("family")[list(TASKS)].sum().to_csv(OUT / "conditional_gain_family.csv")
    plot = result.loc[result.phase == "validation"].copy().sort_values("logloss")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(plot.logloss, np.arange(len(plot)), s=65)
    ax.set_yticks(np.arange(len(plot)), plot.model.astype(str))
    ax.set_xlim(plot.logloss.min() - .003, plot.logloss.max() + .003)
    ax.invert_yaxis()
    ax.set_xlabel("3-task log loss; lower is better")
    ax.set_title("Background, conditional state, and direct models (zoomed)")
    fig.tight_layout()
    fig.savefig(OUT / "03_conditional_comparison.png", dpi=160)
    print("paired test exploratory", comparisons, flush=True)


if __name__ == "__main__":
    main()
