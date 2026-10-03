"""V2 route A: flexible geometric resolution, covariance and soft membership."""

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
from scripts.crypto_predictive_states_stage import CFG, load_panel, period_indices, score_row
from src.crypto.predictive_states import fit_centers, log_loss, make_representation, outcome_labels
from src.crypto.predictive_states_v2 import (center_membership, covariance_geometry,
                                             fit_region_probabilities, region_predict)

OUT = ROOT / "reports/crypto/predictive_states_v2"
COUNTS = (6, 12, 24)
SEED = 20260929


def onehot(ids: np.ndarray, k: int) -> np.ndarray:
    return np.eye(k, dtype=np.float32)[ids]


def estimate_temperature(margin: np.ndarray) -> float:
    return float(max(.03, np.median(margin)))


def evaluate(y: np.ndarray, symbol: np.ndarray, times: pd.Series,
             train_ids: np.ndarray, train_soft: np.ndarray, eval_ids: np.ndarray,
             eval_soft: np.ndarray, k: int, method: str, phase: str) -> dict:
    train = onehot(train_ids, k) if method == "hard" else train_soft
    test = onehot(eval_ids, k) if method == "hard" else eval_soft
    general, local = fit_region_probabilities(y[0], symbol[0], train)
    probabilities = region_predict(test, symbol[1], general, local)
    return score_row(f"{phase}_{method}", phase, y[1], probabilities, times, n_states=k)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    periods = period_indices(panel)
    labels = outcome_labels(panel.return4.to_numpy(), panel.return24.to_numpy(), panel.future_rv4.to_numpy(),
                            panel.rv_slow_15m.to_numpy(), tuple(CFG["return_bin_edges"]))
    symbol = panel.symbol_code.to_numpy(dtype=np.int8)
    rng = np.random.default_rng(SEED)
    fit = np.sort(rng.choice(periods["fit"], size=len(periods["fit"]) // 4, replace=False))
    disc = np.sort(rng.choice(periods["discovery"], size=len(periods["discovery"]) // 4, replace=False))
    geometry = np.concatenate([rng.choice(periods["fit"][symbol[periods["fit"]] == s], 5_000, replace=False)
                               for s in np.unique(symbol)])
    print("loaded", len(panel), "fit", len(fit), "discovery", len(disc), flush=True)
    reps = make_representation(panel, CFG)
    specs = (("X1_equal", reps["X1_endpoints"][:, :-4]),
             ("X2_equal", reps["X2_ordered_path"][:, :-4]))
    records = []
    saved = {}
    for name, x in specs:
        records, saved = experiment(name, x, labels, symbol, panel, periods, fit, disc, geometry, records, saved)
    whiten, matrices = covariance_geometry(reps["X2_ordered_path"][:, :-4], geometry, 16)
    records, saved = experiment("X2_covariance", whiten, labels, symbol, panel, periods,
                                fit, disc, geometry, records, saved, matrices=matrices)
    scores = pd.DataFrame(records)
    scores.to_csv(OUT / "geometry_all_scores.csv", index=False)
    inner = scores.loc[scores.phase == "inner"].sort_values("logloss_day_equal")
    validation = scores.loc[scores.phase == "validation"].sort_values("logloss_day_equal")
    chosen = str(validation.iloc[0].model)
    print("inner top", inner[["model", "logloss"]].head(5).to_dict("records"), flush=True)
    print("validation top", validation[["model", "logloss"]].head(5).to_dict("records"), flush=True)
    winner = saved[chosen]
    if winner["geometry"] == "X1_equal":
        xx = reps["X1_endpoints"][:, :-4]
    elif winner["geometry"] == "X2_equal":
        xx = reps["X2_ordered_path"][:, :-4]
    else:
        xx = whiten
    test = periods["test"]
    ids, distance, soft, margin = center_membership(xx[test], winner["centers"], winner["temperature"])
    weight = onehot(ids, winner["k"]) if winner["method"] == "hard" else soft
    prediction = region_predict(weight, symbol[test], winner["general"], winner["local"])
    test_score = score_row(chosen, "test_exploratory", labels[test], prediction,
                           panel.available_at.iloc[test], n_states=winner["k"])
    scores = pd.concat([scores, pd.DataFrame([{"phase": "test_exploratory", "model": chosen, **test_score}])], ignore_index=True)
    scores.to_csv(OUT / "geometry_all_scores.csv", index=False)
    # Causal prototype and uncertainty cards for the chosen resolution.
    train_ids, train_dist, train_soft, _ = center_membership(xx[disc], winner["centers"], winner["temperature"])
    radius = {k: float(np.quantile(train_dist[train_ids == k], .9)) for k in range(winner["k"])}
    cards = []
    geometry_id = center_membership(xx[geometry], winner["centers"], winner["temperature"])[0]
    for region in range(winner["k"]):
        member = ids == region
        history = train_ids == region
        candidates = geometry[geometry_id == region]
        if len(candidates):
            medoid = candidates[np.argmin(np.sum((xx[candidates] - winner["centers"][region]) ** 2, axis=1))]
            medoid_time = str(panel.available_at.iloc[medoid])
            medoid_symbol = int(symbol[medoid])
        else:
            medoid_time, medoid_symbol = "", -1
        cards.append({"region": region, "discovery_rows_sampled": int(history.sum()),
                      "discovery_days": int(panel.available_at.iloc[disc[history]].dt.floor("D").nunique()),
                      "test_rows": int(member.sum()),
                      "test_days": int(panel.available_at.iloc[test[member]].dt.floor("D").nunique()),
                      "test_symbols": int(np.unique(symbol[test[member]]).size),
                      "test_mean_4h_bp": float(panel.return4.iloc[test[member]].mean() * 1e4),
                      "radius90": radius[region], "medoid_at": medoid_time,
                      "medoid_symbol_code": medoid_symbol})
    pd.DataFrame(cards).to_csv(OUT / "geometry_winner_regions.csv", index=False)
    loss = log_loss(labels[test], prediction)
    rows = panel.iloc[test][["symbol", "available_at"]].copy()
    rows["region"] = ids
    rows["distance"] = distance
    rows["margin"] = margin
    rows["support90"] = np.array([distance[j] <= radius[int(k)] for j, k in enumerate(ids)])
    rows["loss_4h_return"] = loss[:, 0]
    rows["loss_24h_return"] = loss[:, 1]
    rows["loss_4h_vol"] = loss[:, 2]
    rows.to_parquet(OUT / "geometry_winner_test_rows.parquet", index=False, compression="zstd")
    joblib.dump(winner, OUT / "geometry_winner.joblib", compress=3)
    (OUT / "geometry_manifest.json").write_text(json.dumps({"chosen_on": "2025-07 through 2026-01 development validation",
        "test_status": "exploratory reuse of 2026", "winner": chosen, "candidates": len(validation),
        "predefined_k": COUNTS, "geometry_cuda": True}, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(11, 6))
    view = validation.sort_values("logloss").copy()
    ax.scatter(view.logloss, np.arange(len(view)), s=45)
    ax.set_yticks(np.arange(len(view)), view.model.astype(str))
    ax.set_xlim(view.logloss.min() - .001, view.logloss.max() + .001)
    ax.invert_yaxis()
    ax.set_title("Geometry-only candidates: development validation (zoomed)")
    ax.set_xlabel("Mean 3-task log loss; lower is better")
    fig.tight_layout()
    fig.savefig(OUT / "02_geometry_candidates.png", dpi=160)
    print("winner", chosen, "test exploratory", test_score["logloss"], flush=True)


def experiment(name: str, x: np.ndarray, labels: np.ndarray, symbol: np.ndarray, panel: pd.DataFrame,
               periods: dict, fit: np.ndarray, disc: np.ndarray, geometry: np.ndarray,
               records: list, saved: dict, matrices: list[np.ndarray] | None = None):
    for k in COUNTS:
        centers = fit_centers(x[geometry], k, SEED + k)
        # Temperature is learned from input geometry only.
        _, _, _, gap = center_membership(x[geometry], centers, .1)
        temperature = estimate_temperature(gap)
        fit_id, _, fit_soft, _ = center_membership(x[fit], centers, temperature)
        inner = periods["inner"]
        inner_id, _, inner_soft, _ = center_membership(x[inner], centers, temperature)
        disc_id, _, disc_soft, _ = center_membership(x[disc], centers, temperature)
        val = periods["validation"]
        val_id, _, val_soft, _ = center_membership(x[val], centers, temperature)
        for method in ("hard", "soft"):
            model = f"{name}_K{k}_{method}"
            for phase, train_idx, tr_id, tr_soft, eval_idx, ev_id, ev_soft in (
                ("inner", fit, fit_id, fit_soft, inner, inner_id, inner_soft),
                ("validation", disc, disc_id, disc_soft, val, val_id, val_soft)):
                score = evaluate((labels[train_idx], labels[eval_idx]),
                                 (symbol[train_idx], symbol[eval_idx]), panel.available_at.iloc[eval_idx],
                                 tr_id, tr_soft, ev_id, ev_soft, k, method, phase)
                records.append({**score, "model": model, "geometry": name, "k": k,
                                "membership": method, "phase": phase, "temperature": temperature})
            weights = onehot(disc_id, k) if method == "hard" else disc_soft
            general, local = fit_region_probabilities(labels[disc], symbol[disc], weights)
            saved[model] = {"geometry": name, "method": method, "k": k, "centers": centers,
                            "temperature": temperature, "matrices": matrices,
                            "general": general, "local": local}
        print(name, "K", k, "hard/soft validation",
              [(r["model"], round(r["logloss"], 6)) for r in records[-3::2]], flush=True)
    return records, saved


if __name__ == "__main__":
    main()
