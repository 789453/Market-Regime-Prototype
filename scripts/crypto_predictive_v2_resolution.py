"""Adapt geometric resolution beyond the initial grid, retaining all failed K."""

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
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.crypto_predictive_states_stage import CFG, load_panel, paired_block_interval, period_indices, score_row
from src.crypto.predictive_states import fit_centers, log_loss, make_representation, outcome_labels
from src.crypto.predictive_states_v2 import center_membership, fit_region_probabilities, region_predict

OUT = ROOT / "reports/crypto/predictive_states_v2"
COUNTS = (24, 48, 96)
SEED = 20260929


def sample_by_symbol(index: np.ndarray, symbol: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    return np.concatenate([rng.choice(index[symbol[index] == s], size, replace=False)
                           for s in np.unique(symbol)])


def main() -> None:
    panel = load_panel()
    period = period_indices(panel)
    y = outcome_labels(panel.return4.to_numpy(), panel.return24.to_numpy(), panel.future_rv4.to_numpy(),
                       panel.rv_slow_15m.to_numpy(), tuple(CFG["return_bin_edges"]))
    symbol = panel.symbol_code.to_numpy(dtype=np.int8)
    x = make_representation(panel, CFG)["X1_endpoints"][:, :-4]
    rng = np.random.default_rng(SEED)
    geometry = sample_by_symbol(period["fit"], symbol, 5_000, rng)
    fit = np.sort(rng.choice(period["fit"], len(period["fit"]) // 4, replace=False))
    disc = np.sort(rng.choice(period["discovery"], len(period["discovery"]) // 4, replace=False))
    inner, val, test = period["inner"], period["validation"], period["test"]
    rows, daily = [], {}
    saved = {}
    for k in COUNTS:
        centers = fit_centers(x[geometry], k, SEED + k)
        _, _, _, gap = center_membership(x[geometry], centers, .1)
        temperature = max(.03, float(np.median(gap)))
        fs, _, fq, _ = center_membership(x[fit], centers, temperature)
        ins, _, iq, _ = center_membership(x[inner], centers, temperature)
        ds, dd, dq, _ = center_membership(x[disc], centers, temperature)
        vs, vd, vq, _ = center_membership(x[val], centers, temperature)
        gp_fit, sp_fit = fit_region_probabilities(y[fit], symbol[fit], fq)
        gp_disc, sp_disc = fit_region_probabilities(y[disc], symbol[disc], dq)
        for phase, index, weight, general, local in (
            ("inner", inner, iq, gp_fit, sp_fit),
            ("validation", val, vq, gp_disc, sp_disc)):
            p = region_predict(weight, symbol[index], general, local)
            rows.append({"k": k, "phase": phase, "temperature": temperature,
                         **score_row(f"X1_soft_K{k}", phase, y[index], p,
                                     panel.available_at.iloc[index], n_states=k)})
            if phase == "validation":
                daily[k] = log_loss(y[index], p).mean(axis=1)
        count = np.bincount(ds, minlength=k)
        rows[-1].update({"min_sampled_region": int(count.min()),
                         "p10_sampled_region": float(np.quantile(count, .1)),
                         "median_sampled_region": float(np.median(count))})
        saved[k] = {"centers": centers, "temperature": temperature,
                    "general": gp_disc, "local": sp_disc,
                    "sampled_region_counts": count}
        print("K", k, "inner/val", [(r["phase"], round(r["logloss"], 6)) for r in rows[-2:]],
              "min sampled region", count.min(), flush=True)
    score = pd.DataFrame(rows)
    score.to_csv(OUT / "resolution_scores.csv", index=False)
    best_k = int(score.loc[score.phase == "validation"].sort_values("logloss_day_equal").k.iloc[0])
    one_se = []
    for k in COUNTS:
        delta, low, high = paired_block_interval(daily[k], daily[best_k], panel.available_at.iloc[val])
        one_se.append({"k": k, "relative_to_raw_best": best_k,
                       "mean_daily_difference": delta, "lower_7d": low, "upper_7d": high})
    interval = pd.DataFrame(one_se)
    interval.to_csv(OUT / "resolution_paired_validation.csv", index=False)
    conservative = int(interval.loc[interval.lower_7d <= 0].k.min())
    chosen = conservative
    print("raw best", best_k, "smallest within paired interval", chosen, flush=True)
    winner = saved[chosen]
    ids, distance, weight, margin = center_membership(x[test], winner["centers"], winner["temperature"])
    pred = region_predict(weight, symbol[test], winner["general"], winner["local"])
    test_score = score_row(f"X1_soft_K{chosen}", "test_exploratory", y[test], pred,
                           panel.available_at.iloc[test], n_states=chosen)
    score = pd.concat((score, pd.DataFrame([{"k": chosen, "phase": "test_exploratory", **test_score}])), ignore_index=True)
    score.to_csv(OUT / "resolution_scores.csv", index=False)
    frame = panel.iloc[test][["symbol", "available_at"]].copy()
    frame["prototype"] = ids
    frame["distance"] = distance
    frame["margin"] = margin
    loss = log_loss(y[test], pred)
    for j, name in enumerate(("4h_return", "24h_return", "4h_vol")):
        frame[f"{name}_loss"] = loss[:, j]
    frame.to_parquet(OUT / "resolution_chosen_test_rows.parquet", index=False, compression="zstd")
    joblib.dump(winner, OUT / "resolution_chosen_model.joblib", compress=3)
    (OUT / "resolution_manifest.json").write_text(json.dumps({"counts": COUNTS,
        "decision": "smallest K whose paired validation difference versus raw best has interval including zero",
        "raw_best_k": best_k, "chosen_k": chosen,
        "test_status": "exploratory reuse, not independent confirmation"}, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8, 4))
    for phase in ("inner", "validation"):
        sub = score.loc[score.phase == phase]
        ax.plot(sub.k, sub.logloss, marker="o", label=phase)
    ax.set_xlabel("Geometric prototypes K")
    ax.set_ylabel("Three-target distribution log loss")
    ax.legend()
    ax.set_title("Adaptive resolution: no fixed eight-state cap")
    fig.tight_layout()
    fig.savefig(OUT / "04_resolution_curve.png", dpi=160)

    # Identity stability is distinct from outcome performance. Refit 2023 and
    # 2024 geometries; match labels on a common Jan-Jun 2025 input set.
    centers_by_year = []
    for year in (2023, 2024):
        year_idx = period["fit"][panel.available_at.iloc[period["fit"]].dt.year.to_numpy() == year]
        year_geometry = sample_by_symbol(year_idx, symbol, 2_500, rng)
        centers_by_year.append(fit_centers(x[year_geometry], chosen, SEED + year))
    first, second = centers_by_year
    matrix = np.sum((first[:, None, :] - second[None, :, :]) ** 2, axis=2)
    source, destination = linear_sum_assignment(matrix)
    match = np.empty(chosen, dtype=np.int16)
    match[destination] = source
    a, _, _, _ = center_membership(x[inner], first, winner["temperature"])
    b, _, _, _ = center_membership(x[inner], second, winner["temperature"])
    stable = {"chosen_k": chosen, "matched_assignment_agreement": float(np.mean(a == match[b])),
              "adjusted_rand_index": float(adjusted_rand_score(a, b)),
              "mean_matched_center_distance": float(np.sqrt(matrix[source, destination]).mean()),
              "comparison": "2023 vs 2024 independently fitted centers; common 2025H1 paths"}
    (OUT / "resolution_identity_stability.json").write_text(json.dumps(stable, indent=2), encoding="utf-8")
    print("test exploratory", test_score["logloss"], "identity", stable, flush=True)


if __name__ == "__main__":
    main()
