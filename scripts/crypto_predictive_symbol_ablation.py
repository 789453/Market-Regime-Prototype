"""Matched X1 pooled-vs-symbol percentile reference diagnostic."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.crypto_predictive_states_stage import CFG, SCOPE, load_panel, period_indices, score_row
from src.crypto.predictive_states import fit_states, log_loss, make_representation, outcome_labels, predict_states


def main() -> None:
    panel = load_panel()
    periods = period_indices(panel)
    symbol = panel.symbol_code.to_numpy(dtype=np.int8)
    y = outcome_labels(panel.return4.to_numpy(), panel.return24.to_numpy(), panel.future_rv4.to_numpy(),
                       panel.rv_slow_15m.to_numpy(), tuple(CFG["return_bin_edges"]))
    rng = np.random.default_rng(CFG["random_seed"])
    chosen = {}
    for key in ("fit", "inner", "discovery"):
        idx = periods[key]
        chosen[key] = np.sort(rng.choice(idx, size=max(1, len(idx) // CFG["outcome_sample_stride"]), replace=False))
    x = make_representation(panel, CFG, pooled=True)["X1_endpoints"]
    model = fit_states(x, y, symbol, panel.available_at, chosen["fit"], chosen["inner"], chosen["discovery"], CFG)
    results = []
    concentration = []
    daily_loss = []
    for period in ("validation", "test"):
        idx = periods[period]
        p, state, _, accepted = predict_states(model, x[idx], symbol[idx], split=True)
        results.append(score_row("X1_pooled_split_exploratory", period, y[idx], p,
                                 panel.available_at.iloc[idx], accepted, len(np.unique(state))))
        daily = pd.DataFrame({"date": panel.available_at.iloc[idx].dt.floor("D").to_numpy(),
                              "loss": log_loss(y[idx], p).mean(axis=1)}).groupby("date").loss.mean().reset_index()
        daily["period"] = period
        daily_loss.append(daily)
        for leaf in np.unique(state):
            sub = symbol[idx][(state == leaf) & accepted]
            if len(sub):
                concentration.append({"period": period, "leaf": int(leaf), "rows": len(sub),
                                      "largest_symbol_share": float(np.bincount(sub, minlength=12).max() / len(sub))})
    pd.DataFrame(results).to_csv(SCOPE / "pooled_X1_scores_exploratory.csv", index=False)
    pd.DataFrame(concentration).to_csv(SCOPE / "pooled_X1_symbol_concentration.csv", index=False)
    pd.concat(daily_loss).to_csv(SCOPE / "pooled_X1_daily_loss.csv", index=False)
    print(pd.DataFrame(results).to_string(index=False))
    print(pd.DataFrame(concentration).groupby("period").largest_symbol_share.describe().to_string())


if __name__ == "__main__":
    main()
