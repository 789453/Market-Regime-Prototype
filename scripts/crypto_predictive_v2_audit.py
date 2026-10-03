"""Read-only audit of v1 information loss before changing its architecture."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "reports/crypto/predictive_states_v1"
OUT = ROOT / "reports/crypto/predictive_states_v2"
TASKS = ("4h_return", "24h_return", "4h_vol")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    columns = ["symbol", "available_at", "state", "accepted", "geometry_distance",
               "state_logloss", "direct_logloss", "background_logloss"]
    columns += [f"{model}_{task}_logloss" for model in ("state", "direct", "background") for task in TASKS]
    frame = pq.read_table(OLD / "test_predictions.parquet", columns=columns).to_pandas()
    frame["month"] = frame.available_at.dt.strftime("%Y-%m")
    frame["gap_direct"] = frame.state_logloss - frame.direct_logloss
    frame["gain_background"] = frame.background_logloss - frame.state_logloss
    for group, name in ((["state"], "leaf"), (["symbol"], "symbol"), (["month"], "month"),
                        (["state", "month"], "leaf_month")):
        result = frame.groupby(group, observed=True).agg(
            rows=("gap_direct", "size"), state_loss=("state_logloss", "mean"),
            direct_loss=("direct_logloss", "mean"), background_loss=("background_logloss", "mean"),
            gap_direct=("gap_direct", "mean"), gain_background=("gain_background", "mean"),
            accepted=("accepted", "mean")).reset_index()
        result.to_csv(OUT / f"audit_by_{name}.csv", index=False)
    task_rows = []
    for task in TASKS:
        for model in ("state", "direct", "background"):
            task_rows.append({"task": task, "model": model,
                              "logloss": float(frame[f"{model}_{task}_logloss"].mean())})
    pd.DataFrame(task_rows).to_csv(OUT / "audit_by_task.csv", index=False)

    # Geometry support and prediction reliability are separate hypotheses.
    frame["distance_rank_in_state"] = frame.groupby("state").geometry_distance.rank(pct=True)
    frame["distance_band"] = pd.cut(frame.distance_rank_in_state,
                                    [0, .2, .4, .6, .8, 1.00001], labels=["0-20", "20-40", "40-60", "60-80", "80-100"])
    support = frame.groupby("distance_band", observed=True).agg(
        rows=("gap_direct", "size"), distance=("geometry_distance", "mean"),
        state_loss=("state_logloss", "mean"), direct_loss=("direct_logloss", "mean"),
        background_loss=("background_logloss", "mean"), gap_direct=("gap_direct", "mean"),
        gain_background=("gain_background", "mean"), accepted=("accepted", "mean")).reset_index()
    support.to_csv(OUT / "audit_center_support.csv", index=False)
    graph = pq.read_table(OLD / "neighbor_graph_predictions.parquet",
                          columns=["symbol", "available_at", "k8_distance", "accepted", "logloss"]).to_pandas()
    graph = graph.loc[graph.available_at.isin(frame.available_at.unique())].copy()
    merged = frame[["symbol", "available_at", "state_logloss", "direct_logloss", "background_logloss"]].merge(
        graph, on=["symbol", "available_at"], how="inner", validate="one_to_one")
    merged["neighbor_rank"] = merged.k8_distance.rank(pct=True)
    merged["neighbor_band"] = pd.cut(merged.neighbor_rank, [0, .2, .4, .6, .8, 1.00001],
                                     labels=["0-20", "20-40", "40-60", "60-80", "80-100"])
    merged.groupby("neighbor_band", observed=True).agg(
        rows=("logloss", "size"), k8_distance=("k8_distance", "mean"),
        graph_loss=("logloss", "mean"), state_loss=("state_logloss", "mean"),
        direct_loss=("direct_logloss", "mean"), background_loss=("background_logloss", "mean"),
        accepted=("accepted", "mean")).reset_index().to_csv(OUT / "audit_real_neighbor_support.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    task_table = pd.DataFrame(task_rows).pivot(index="task", columns="model", values="logloss")
    task_table[["background", "state", "direct"]].plot.bar(ax=axes[0, 0], rot=0)
    axes[0, 0].set_title("Distribution loss by target")
    axes[0, 0].set_ylabel("Natural log loss; lower is better")
    monthly = pd.read_csv(OUT / "audit_by_month.csv")
    axes[0, 1].plot(monthly.month, monthly.gap_direct, marker="o", label="state - direct")
    axes[0, 1].plot(monthly.month, monthly.gain_background, marker="o", label="background - state")
    axes[0, 1].tick_params(axis="x", rotation=40)
    axes[0, 1].legend()
    axes[0, 1].set_title("Monthly information gains")
    axes[1, 0].plot(support.distance_band.astype(str), support.gap_direct, marker="o", label="state - direct")
    axes[1, 0].plot(support.distance_band.astype(str), support.gain_background, marker="o", label="background - state")
    axes[1, 0].legend()
    axes[1, 0].set_title("Center distance percentile within leaf")
    leaves = pd.read_csv(OUT / "audit_by_leaf.csv").sort_values("gap_direct")
    axes[1, 1].bar(leaves.state.astype(str), leaves.gap_direct)
    axes[1, 1].set_title("State - direct loss by leaf")
    axes[1, 1].tick_params(axis="x", rotation=55)
    fig.tight_layout()
    fig.savefig(OUT / "01_information_loss_audit.png", dpi=160)
    print("audit rows", len(frame), "graph matched", len(merged), flush=True)


if __name__ == "__main__":
    main()
