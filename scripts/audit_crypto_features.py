"""Summarize availability and redundancy of the generated development features."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.crypto.features import GROUPS


SOURCE = ROOT / "data/crypto/features"
OUT = ROOT / "reports/crypto/feature_group_audit.csv"


def main() -> None:
    rows = []
    for path in sorted(SOURCE.glob("*USDT.parquet")):
        data = pd.read_parquet(path)
        data = data.loc[data.available_at < "2026-01-01"].iloc[1000:]
        for frequency in ("5m", "15m"):
            for group, names in GROUPS.items():
                columns = [f"{name}_{frequency}" for name in names]
                values = data[columns]
                sampled = values.iloc[::max(len(values) // 10000, 1)]
                corr = sampled.corr().abs().to_numpy()
                upper = corr[np.triu_indices(len(columns), 1)]
                rows.append({
                    "symbol": path.stem, "frequency": frequency, "group": group,
                    "min_nonnull_fraction": float(values.notna().mean().min()),
                    "median_abs_pair_corr": float(np.nanmedian(upper)),
                    "pairs_abs_corr_gt_095": int(np.nansum(upper > 0.95)),
                })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(pd.DataFrame(rows).groupby(["frequency", "group"]).agg(
        min_nonnull=("min_nonnull_fraction", "min"),
        median_pair_corr=("median_abs_pair_corr", "median"),
        high_corr_pairs=("pairs_abs_corr_gt_095", "sum"),
    ).to_string())


if __name__ == "__main__":
    main()
