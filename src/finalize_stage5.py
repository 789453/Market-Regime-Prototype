"""Finalize Stage 5 statistical and mechanical audit artifacts."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from src.fusion import cluster_detectors, detector_dependence, effective_detector_count
from src.metrics import deflated_sharpe_ratio
from src.run_stage5 import DETECTORS, FIGS, PORTFOLIOS, PROCESSED, TABLES, TRADABLE, _load


K_FINAL = 214


def main() -> None:
    bars, context, _, signals, _, _ = _load()
    signals = {name: frame.loc[frame.index.get_level_values("root").isin(TRADABLE)] for name, frame in signals.items()}
    corr, common = detector_dependence(signals)
    clusters = cluster_detectors(corr)
    corr.to_csv(TABLES / "05_detector_direction_correlation.csv")
    common.to_csv(TABLES / "05_detector_common_trigger_rate.csv")
    dependency = pd.DataFrame(
        {"detector": DETECTORS, "cluster": clusters.reindex(DETECTORS).to_numpy()}
    )
    dependency["effective_detector_count"] = effective_detector_count(corr.to_numpy())
    dependency.to_csv(TABLES / "05_detector_clusters.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    image = axes[0].imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    axes[0].set_xticks(range(len(DETECTORS)), DETECTORS, rotation=45)
    axes[0].set_yticks(range(len(DETECTORS)), DETECTORS)
    axes[0].set_title("Joint-trigger direction correlation")
    fig.colorbar(image, ax=axes[0], fraction=.046)
    distance = 1 - np.clip(np.abs(corr.to_numpy()), 0, 1)
    dendrogram(linkage(squareform(distance, checks=False), method="complete"), labels=DETECTORS, ax=axes[1])
    axes[1].axhline(.4, color="#b43c39", ls="--", label="|rho| = 0.60")
    axes[1].set(title="Complete-linkage redundancy tree", ylabel="1 - |rho|")
    axes[1].legend()
    fig.tight_layout(); fig.savefig(FIGS / "05_detector_redundancy_all.png", dpi=150); plt.close(fig)

    metrics_path = TABLES / "05_portfolio_metrics.csv"
    metrics = pd.read_csv(metrics_path).set_index("portfolio")
    audit_rows = []
    for name in PORTFOLIOS:
        daily = pd.read_csv(TABLES / f"05_{name}_daily_returns.csv", index_col=0)["net_return"]
        metrics.loc[name, ["sr0", "deflated_sharpe_probability"]] = pd.Series(
            deflated_sharpe_ratio(daily, K_FINAL)
        )
        bt = pd.read_parquet(PROCESSED / f"backtest_stage5_{name}.parquet")
        positive = bt.position.clip(lower=0).groupby(level="timestamp").sum()
        negative = (-bt.position.clip(upper=0)).groupby(level="timestamp").sum()
        audit_rows.append(
            {
                "portfolio": name,
                "net_identity_max_abs_error": float((bt.net_return - (bt.gross_return - bt.cost)).abs().max()),
                "max_abs_instrument_position": float(bt.position.abs().max()),
                "max_same_direction_long": float(positive.max()),
                "max_same_direction_short": float(negative.max()),
                "min_cost": float(bt.cost.min()),
                "min_executed_delay_bars": float(bt.loc[bt.executed, "execution_delay_bars"].min()),
                "max_executed_delay_bars": float(bt.loc[bt.executed, "execution_delay_bars"].max()),
                "contains_hti": bool("HTI" in bt.index.get_level_values("root")),
                "row_count": len(bt),
                "finite_net_fraction": float(np.isfinite(bt.net_return).mean()),
            }
        )
    metrics.to_csv(metrics_path)
    pd.DataFrame(audit_rows).to_csv(TABLES / "05_mechanical_audit.csv", index=False)

    manifest_path = PROCESSED / "stage5_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "global_k_after_stage5": K_FINAL,
            "stage5_candidate_variants": 46,
            "holdout_views_used": 2,
            "authoritative_shuffle_files": [
                f"05_{name}_shuffle_500_exitfix.csv" for name in PORTFOLIOS
            ],
            "superseded_shuffle_files": [f"05_{name}_shuffle_500.csv" for name in PORTFOLIOS],
            "tests_passed": 41,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(dependency.to_string(index=False))
    print(pd.DataFrame(audit_rows).to_string(index=False))
    print(metrics[["sharpe", "sharpe_ci_low", "sharpe_ci_high", "sr0", "deflated_sharpe_probability"]].to_string())


if __name__ == "__main__":
    main()
