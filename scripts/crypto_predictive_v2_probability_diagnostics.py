"""Probability calibration, sharpness and ordinal score on identical test rows."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/crypto/predictive_states_v2"
OLD = ROOT / "reports/crypto/predictive_states_v1"


def expected_calibration(p: np.ndarray, y: np.ndarray) -> tuple[float, pd.DataFrame]:
    band = np.minimum(np.floor(p * 10).astype(int), 9)
    rows = []
    for j in range(10):
        mask = band == j
        if mask.any():
            rows.append({"bin": j, "rows": int(mask.sum()),
                         "forecast": float(p[mask].mean()), "observed": float(y[mask].mean())})
    table = pd.DataFrame(rows)
    error = float(np.average(abs(table.forecast - table.observed), weights=table.rows))
    return error, table


def ranked_probability_score(p: np.ndarray, label: np.ndarray) -> float:
    cumulative = p.cumsum(axis=1)[:, :-1]
    actual = (label[:, None] <= np.arange(p.shape[1] - 1)[None, :]).astype(float)
    return float(np.mean((cumulative - actual) ** 2))


def main() -> None:
    old = pd.read_parquet(OLD / "test_predictions.parquet")
    new = pd.read_parquet(OUT / "conditional_test_predictions.parquet")
    if not np.array_equal(old.available_at.to_numpy(), new.available_at.to_numpy()):
        raise AssertionError("probability audit requires aligned observations")
    tasks = {"4h_return": (5, old.target_4h_bin.to_numpy()),
             "24h_return": (5, old.target_24h_bin.to_numpy()),
             "4h_vol": (2, old.target_high_rv4.to_numpy())}
    source = {"v1_state": (old, "state"), "v1_direct": (old, "direct"),
              "v1_background": (old, "background"),
              "market_clock_rv_symbol": (new, "market_clock_background"),
              "compact_no_prototype": (new, "compact_no_prototype"),
              "compact_v1_prototype": (new, "compact_with_prototype")}
    rows, reliability = [], []
    for name, (frame, prefix) in source.items():
        for task, (classes, label) in tasks.items():
            p = frame[[f"{prefix}_{task}_p{j}" for j in range(classes)]].to_numpy()
            if task == "4h_vol":
                forecast = p[:, 1]
                ece, table = expected_calibration(forecast, label)
                table["model"] = name
                reliability.append(table)
                brier = float(np.mean((forecast - label) ** 2))
                sharpness = float(np.std(forecast))
            else:
                ece, brier, sharpness = np.nan, np.nan, float(np.mean(np.std(p, axis=0)))
            rows.append({"model": name, "task": task,
                         "logloss": float(np.mean(-np.log(np.maximum(p[np.arange(len(label)), label], 1e-8)))),
                         "ranked_probability_score": ranked_probability_score(p, label) if classes == 5 else np.nan,
                         "brier": brier, "ECE_10bin": ece, "probability_sharpness_std": sharpness})
    pd.DataFrame(rows).to_csv(OUT / "probability_quality.csv", index=False)
    rel = pd.concat(reliability, ignore_index=True)
    rel.to_csv(OUT / "high_vol_reliability.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in ("v1_state", "v1_direct", "market_clock_rv_symbol", "compact_no_prototype"):
        sub = rel.loc[rel.model == name].sort_values("forecast")
        ax.plot(sub.forecast, sub.observed, marker="o", label=name)
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    ax.set_xlim(.1, .9)
    ax.set_ylim(.1, .9)
    ax.set_xlabel("Predicted probability of high 4h volatility")
    ax.set_ylabel("Observed event fraction")
    ax.set_title("Calibration on reused 2026 observations")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "10_high_vol_calibration.png", dpi=160)
    print(pd.DataFrame(rows).loc[lambda d: d.task == "4h_vol"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
