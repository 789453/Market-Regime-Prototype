"""Freeze empirical maps and reproduce the selected v2 probability model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.crypto_predictive_states_stage import CFG, load_panel, period_indices, score_row
from src.crypto.predictive_states import fit_reference_maps, outcome_labels
from src.crypto.predictive_states_v2 import infer_compact_geometry

OUT = ROOT / "reports/crypto/predictive_states_v2"


def main() -> None:
    panel = load_panel()
    references = fit_reference_maps(panel, CFG)
    joblib.dump(references, OUT / "frozen_feature_maps.joblib", compress=3)
    conditional = joblib.load(OUT / "hybrid_compact_with_geometry_margin.joblib")
    geometry = joblib.load(OUT / "resolution_chosen_model.joblib")
    rows = period_indices(panel)["test"]
    p, context = infer_compact_geometry(panel, CFG, references, conditional, geometry, rows)
    y = outcome_labels(panel.return4.to_numpy()[rows], panel.return24.to_numpy()[rows],
                       panel.future_rv4.to_numpy()[rows], panel.rv_slow_15m.to_numpy()[rows],
                       tuple(CFG["return_bin_edges"]))
    score = score_row("v2_compact_geometry", "test_exploratory", y, p,
                      panel.available_at.iloc[rows])
    scored = pd.read_csv(OUT / "hybrid_scores.csv")
    target = float(scored.loc[(scored.model == "compact_with_geometry_margin") &
                              (scored.phase == "test_exploratory"), "logloss"].iloc[0])
    if abs(score["logloss"] - target) > 2e-6:
        raise AssertionError(f"saved model and frozen map disagree: {score['logloss']} != {target}")
    if not all(np.allclose(task.sum(axis=1), 1, atol=1e-5) for task in p):
        raise AssertionError("probabilities must be normalized")
    # A streaming caller only needs the last 17 completed snapshots per asset.
    # Compare its final output to the same rows evaluated with the full panel.
    last_at = panel.available_at.iloc[rows[-1]]
    target_rows = np.flatnonzero((panel.available_at == last_at).to_numpy())
    if len(target_rows) != 12 or not np.all(np.isin(target_rows, rows)):
        raise AssertionError("expected 12 synchronized evaluation snapshots")
    window_rows = np.concatenate([np.arange(row - 16, row + 1) for row in target_rows])
    small_panel = panel.iloc[window_rows].reset_index(drop=True)
    small_rows = np.arange(16, len(small_panel), 17)
    short_p, _ = infer_compact_geometry(small_panel, CFG, references, conditional, geometry, small_rows)
    in_test = np.searchsorted(rows, target_rows)
    prefix_difference = max(float(np.max(np.abs(short_p[j] - p[j][in_test]))) for j in range(3))
    if prefix_difference > 2e-6:
        raise AssertionError(f"17-snapshot inference disagrees with full history: {prefix_difference}")
    result = {"model": "compact 29 + symbol 12 + K48 distance/margin 2; three 75-tree/7-leaf heads",
              "feature_reference_fit_end": CFG["discovery_fit_end"],
              "requires": "aligned completed 15m feature snapshots of all 12 symbols and 16 prior steps",
              "inference_uses_future_labels": False,
              "test_status": "exploratory reuse; not a new independent holdout",
              "rows_reproduced": len(rows), "logloss_reproduced": score["logloss"],
              "reference_logloss": target, "geometry_cuda": True,
              "streaming_17_snapshot_max_probability_difference": prefix_difference,
              "probability_tasks": ["4h standardized return five bins", "24h standardized return five bins",
                                    "4h variance above past-24h-equivalent threshold"]}
    (OUT / "inference_manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
