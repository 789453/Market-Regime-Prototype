"""Time-block uncertainty for full-pattern strategy comparisons."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/crypto"


def interval(values: np.ndarray, *, seed: int = 20260928) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    starts = rng.integers(0, n, size=(2000, int(np.ceil(n / 7))))
    positions = ((starts[:, :, None] + np.arange(7)) % n).reshape(2000, -1)[:, :n]
    means = values[positions].mean(axis=1) * 10000
    return tuple(np.quantile(means, [.025, .975]).astype(float))


def main() -> None:
    daily = pd.read_csv(REPORT / "full_pattern_daily.csv", parse_dates=["date"])
    focus = (
        "semantic_seed_fixed4h", "full_equal_fixed4h", "full_redundancy_fixed4h",
        "full_dynamic_fixed4h", "semantic_gated_full_fixed4h", "full_cosine_fixed4h",
        "full_dynamic_sparse_fixed4h", "full_dynamic_hysteresis",
        "full_dynamic_vol_scaled_fixed4h", "semantic_gated_full_train_top1_fixed4h",
        "semantic_gated_full_train_top2_fixed4h", "buyhold",
    )
    rows = []
    for (strategy, cost, year), group in daily.groupby(["strategy", "cost_per_side_bp", daily.date.dt.year]):
        if strategy not in focus or cost not in (0, 4):
            continue
        values = group.sort_values("date").daily_logreturn.to_numpy()
        low, high = interval(values)
        rows.append({"strategy": strategy, "cost_per_side_bp": cost, "year": year,
                     "days": len(values), "mean_daily_bp": float(values.mean() * 10000),
                     "block7_low_bp": low, "block7_high_bp": high,
                     "cumulative_return": float(np.exp(values.sum()) - 1)})
    buyhold = daily.loc[daily.strategy == "buyhold", ["date", "daily_logreturn"]].set_index("date").daily_logreturn
    for strategy in focus:
        if strategy == "buyhold":
            continue
        part = daily.loc[(daily.strategy == strategy) & (daily.cost_per_side_bp == 4), ["date", "daily_logreturn"]].set_index("date").daily_logreturn
        aligned = pd.concat([part, buyhold], axis=1).dropna()
        if aligned.empty:
            continue
        difference = (aligned.iloc[:, 0] - aligned.iloc[:, 1]).to_numpy()
        low, high = interval(difference)
        rows.append({"strategy": strategy + "_minus_buyhold", "cost_per_side_bp": 4, "year": 0,
                     "days": len(difference), "mean_daily_bp": float(difference.mean() * 10000),
                     "block7_low_bp": low, "block7_high_bp": high, "cumulative_return": np.nan})
    result = pd.DataFrame(rows)
    result.to_csv(REPORT / "full_pattern_year_block.csv", index=False)
    print(result.loc[result.strategy.isin(("full_dynamic_fixed4h", "semantic_gated_full_fixed4h",
                                               "semantic_gated_full_train_top1_fixed4h", "buyhold"))].to_string(index=False))


if __name__ == "__main__":
    main()
