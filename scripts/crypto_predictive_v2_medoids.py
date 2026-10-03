"""Actual historical exemplars for the selected geometric regions."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.crypto_predictive_states_stage import CFG, SYMBOLS, load_panel, period_indices
from src.crypto.predictive_states import make_representation
from src.crypto.predictive_states_v2 import center_membership

OUT = ROOT / "reports/crypto/predictive_states_v2"


def main() -> None:
    panel = load_panel()
    period = period_indices(panel)
    x = make_representation(panel, CFG)["X1_endpoints"][:, :-4]
    model = joblib.load(OUT / "resolution_chosen_model.joblib")
    rng = np.random.default_rng(CFG["random_seed"])
    symbol = panel.symbol_code.to_numpy(dtype=np.int8)
    candidates = np.concatenate([rng.choice(period["fit"][symbol[period["fit"]] == s], 3_000, replace=False)
                                 for s in range(len(SYMBOLS))])
    ids, distance, _, _ = center_membership(x[candidates], model["centers"], model["temperature"])
    exemplar = []
    for state in range(len(model["centers"])):
        local = np.flatnonzero(ids == state)
        if not len(local):
            continue
        best = candidates[local[np.argmin(distance[local])]]
        exemplar.append({"prototype": state, "row": int(best), "symbol": SYMBOLS[int(symbol[best])],
                         "available_at": panel.available_at.iloc[best],
                         "center_distance": float(distance[local].min()),
                         "historical_high_vol_p": float(model["general"][2][state, 1])})
    frame = pd.DataFrame(exemplar)
    raw_by_symbol = {}
    for s in frame.symbol.unique():
        raw_by_symbol[s] = pq.read_table(Path(CFG["source"]) / s / "15m.parquet",
                                         columns=["open_time", "close"]).to_pandas()
    paths = []
    for row in frame.itertuples():
        t = int(row.available_at.timestamp() * 1000)
        raw = raw_by_symbol[row.symbol]
        history = raw.loc[(raw.open_time >= t - 4 * 3_600_000) & (raw.open_time < t), "close"].to_numpy()
        if len(history) != 16:
            raise AssertionError(f"expected 16 completed historical bars for {row.symbol} {row.available_at}")
        bp = (np.log(history) - np.log(history[0])) * 10_000
        paths.append(bp)
    values = np.vstack(paths)
    for j in range(16):
        frame[f"past_close_bp_{j}"] = values[:, j]
    frame.to_csv(OUT / "K48_real_medoids.csv", index=False)
    rank = frame.sort_values("historical_high_vol_p")
    selected = pd.concat((rank.head(2), rank.iloc[len(rank)//2 - 1:len(rank)//2 + 1], rank.tail(2)))
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for ax, row in zip(axes.flat, selected.itertuples()):
        y = np.array([getattr(row, f"past_close_bp_{j}") for j in range(16)])
        ax.plot(np.arange(16) / 4, y, marker=".", linewidth=1)
        ax.axhline(0, color="grey", linewidth=.8)
        ax.set_title(f"Region {row.prototype}: {row.symbol} {row.available_at:%Y-%m-%d}\n"
                     f"Historical P(high 4h vol)={row.historical_high_vol_p:.2f}")
        ax.set_ylabel("Past close path, bp from first point")
        ax.set_xlabel("Hours across completed 4h history")
    fig.tight_layout()
    fig.savefig(OUT / "08_actual_historical_paths.png", dpi=160)
    print("actual medoids", len(frame), "all with 16-bar causal paths", flush=True)


if __name__ == "__main__":
    main()
