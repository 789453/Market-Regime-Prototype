"""Year and block-uncertainty diagnostics for the frozen first event pilot."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/crypto"


def block_interval(values: np.ndarray, *, seed: int = 20260928, block: int = 7) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    starts = rng.integers(0, n, size=(2000, int(np.ceil(n / block))))
    idx = ((starts[:, :, None] + np.arange(block)) % n).reshape(2000, -1)[:, :n]
    means = values[idx].mean(axis=1) * 10000
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    daily = pd.read_csv(REPORT / "event_walkforward_daily.csv", parse_dates=["date"])
    daily = daily.loc[daily.side_cost_bp == 10].copy()
    daily["year"] = daily.date.dt.year
    rows = []
    for (strategy, year), group in daily.groupby(["strategy", "year"]):
        returns = group.daily_return.to_numpy()
        lo, hi = block_interval(returns)
        rows.append({
            "strategy": strategy, "year": year, "days": len(returns),
            "mean_daily_bp": float(returns.mean() * 10000),
            "block7_low_bp": lo, "block7_high_bp": hi,
            "year_cumulative_return": float(np.prod(1 + returns) - 1),
        })
    panel = daily.pivot(index="date", columns="strategy", values="daily_return").dropna()
    for strategy in ("target_position", "hysteresis_position"):
        diff = (panel[strategy] - panel.buyhold).to_numpy()
        lo, hi = block_interval(diff)
        rows.append({
            "strategy": f"{strategy}_minus_buyhold", "year": 0, "days": len(diff),
            "mean_daily_bp": float(diff.mean() * 10000),
            "block7_low_bp": lo, "block7_high_bp": hi,
            "year_cumulative_return": np.nan,
        })
    result = pd.DataFrame(rows)
    result.to_csv(REPORT / "event_year_and_block.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
