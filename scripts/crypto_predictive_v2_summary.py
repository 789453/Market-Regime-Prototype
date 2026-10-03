"""Stage cards, paired comparisons and readable figures for the v2 study."""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/crypto/predictive_states_v2"
OLD = ROOT / "reports/crypto/predictive_states_v1"


def align(left: pd.DataFrame, right: pd.DataFrame) -> None:
    if not (np.array_equal(left.symbol.to_numpy(), right.symbol.to_numpy()) and
            np.array_equal(left.available_at.to_numpy(), right.available_at.to_numpy())):
        raise AssertionError("model evaluations must use identical symbol/time rows")


def main() -> None:
    old = pd.read_parquet(OLD / "test_predictions.parquet",
                          columns=["symbol", "available_at", "return4", "return24", "target_high_rv4",
                                   "state_logloss", "direct_logloss", "background_logloss"])
    geo = pd.read_parquet(OUT / "geometry_winner_test_rows.parquet")
    res = pd.read_parquet(OUT / "resolution_chosen_test_rows.parquet")
    cond = pd.read_parquet(OUT / "conditional_test_predictions.parquet")
    hybrid_margin = pd.read_parquet(OUT / "hybrid_compact_with_geometry_margin_test_rows.parquet")
    hybrid_soft = pd.read_parquet(OUT / "hybrid_compact_with_soft_regions_test_rows.parquet")
    for other in (geo, res, cond, hybrid_margin, hybrid_soft):
        align(old, other)
    losses = pd.DataFrame({"symbol": old.symbol, "available_at": old.available_at,
                           "month": old.available_at.dt.strftime("%Y-%m"),
                           "v1_state": old.state_logloss,
                           "v1_direct": old.direct_logloss,
                           "v1_symbol_background": old.background_logloss,
                           "K24_soft": geo[["loss_4h_return", "loss_24h_return", "loss_4h_vol"]].mean(axis=1),
                           "K48_soft": res[["4h_return_loss", "24h_return_loss", "4h_vol_loss"]].mean(axis=1),
                           "market_clock_rv_symbol": cond[[f"market_clock_background_{x}_loss" for x in
                                      ("4h_return", "24h_return", "4h_vol")]].mean(axis=1),
                           "compact_no_prototype": cond[[f"compact_no_prototype_{x}_loss" for x in
                                      ("4h_return", "24h_return", "4h_vol")]].mean(axis=1),
                           "compact_v1_prototype": cond[[f"compact_with_prototype_{x}_loss" for x in
                                      ("4h_return", "24h_return", "4h_vol")]].mean(axis=1),
                           "compact_K48_distance": hybrid_margin.loss,
                           "compact_K48_soft": hybrid_soft.loss})
    names = list(losses.columns[3:])
    losses.groupby("month")[names].mean().reset_index().to_csv(OUT / "all_method_monthly_test_exploratory.csv", index=False)
    results = []
    days = losses.available_at.dt.floor("D").to_numpy()
    daily = losses.groupby(losses.available_at.dt.floor("D"))[names].mean().reset_index(drop=True)
    rng = np.random.default_rng(20260929)
    blocks = ((rng.integers(0, len(daily), size=(2000, int(np.ceil(len(daily) / 7))))[:, :, None]
               + np.arange(7)) % len(daily)).reshape(2000, -1)[:, :len(daily)]
    for name in names:
        for reference in ("v1_state", "v1_direct", "compact_no_prototype"):
            if name == reference:
                continue
            diff = daily[name].to_numpy() - daily[reference].to_numpy()
            sample = diff[blocks].mean(axis=1)
            results.append({"model": name, "minus": reference, "daily_difference": float(diff.mean()),
                            "lower_7d": float(np.quantile(sample, .025)),
                            "upper_7d": float(np.quantile(sample, .975))})
    pd.DataFrame(results).to_csv(OUT / "all_method_paired_test_exploratory.csv", index=False)

    # K48 conditional state cards: future market is used only here for ex-post attribution.
    market_sum = old.groupby("available_at").return4.transform("sum")
    loo_residual = old.return4 - (market_sum - old.return4) / 11
    card = pd.DataFrame({"prototype": res.prototype, "available_at": old.available_at,
                         "symbol": old.symbol, "return4": old.return4, "return24": old.return24,
                         "loo_residual": loo_residual, "high_vol": old.target_high_rv4,
                         "loss": losses.K48_soft, "distance": res.distance})
    card["day"] = card.available_at.dt.floor("D")
    card["month"] = card.available_at.dt.strftime("%Y-%m")
    result = card.groupby("prototype").agg(
        rows=("return4", "size"), days=("day", "nunique"), months=("month", "nunique"),
        symbols=("symbol", "nunique"), mean_4h_bp=("return4", lambda s: float(s.mean() * 1e4)),
        mean_24h_bp=("return24", lambda s: float(s.mean() * 1e4)),
        mean_loo_residual_bp=("loo_residual", lambda s: float(s.mean() * 1e4)),
        high_vol_rate=("high_vol", "mean"), test_loss=("loss", "mean"),
        mean_distance=("distance", "mean")).reset_index()
    model = joblib.load(OUT / "resolution_chosen_model.joblib")
    result["discovery_sampled_rows"] = model["sampled_region_counts"][result.prototype]
    for task, label in enumerate(("return4", "return24", "vol4")):
        for category in range(model["general"][task].shape[1]):
            result[f"historical_{label}_p{category}"] = model["general"][task][result.prototype, category]
    result.to_csv(OUT / "K48_prototype_distribution_cards.csv", index=False)

    rank = card.groupby("prototype").distance.rank(pct=True)
    card["distance_band"] = pd.cut(rank, [0, .2, .4, .6, .8, 1.00001],
                                    labels=["0-20", "20-40", "40-60", "60-80", "80-100"])
    card["direct_loss"] = old.direct_logloss
    card["compact_loss"] = losses.compact_no_prototype
    card.groupby("distance_band", observed=True).agg(
        rows=("loss", "size"), state_loss=("loss", "mean"),
        direct_loss=("direct_loss", "mean"), compact_loss=("compact_loss", "mean"),
        mean_distance=("distance", "mean")).reset_index().to_csv(
            OUT / "K48_coverage_error.csv", index=False)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    selected = ["v1_state", "K24_soft", "K48_soft", "market_clock_rv_symbol",
                "compact_no_prototype", "compact_K48_distance", "v1_direct"]
    y = [losses[name].mean() for name in selected]
    ax[0, 0].scatter(y, selected, s=65)
    ax[0, 0].axvline(losses.v1_direct.mean(), color="grey", linestyle="--", linewidth=1)
    ax[0, 0].set_xlim(1.215, 1.27)
    ax[0, 0].invert_yaxis()
    ax[0, 0].set_title("2026 reuse: method-level distribution loss (zoomed)")
    ax[0, 0].set_xlabel("Mean log loss; lower is better")
    month = losses.groupby("month")[names].mean()
    for name in ("K48_soft", "compact_no_prototype", "compact_K48_distance"):
        ax[0, 1].plot(month.index, month[name] - month.v1_direct, marker="o", label=name)
    ax[0, 1].axhline(0, color="black", linewidth=1)
    ax[0, 1].tick_params(axis="x", rotation=40)
    ax[0, 1].legend(fontsize=8)
    ax[0, 1].set_title("Monthly loss minus v1 direct")
    cover = pd.read_csv(OUT / "K48_coverage_error.csv")
    for name in ("state_loss", "compact_loss", "direct_loss"):
        ax[1, 0].plot(cover.distance_band, cover[name], marker="o", label=name)
    ax[1, 0].legend()
    ax[1, 0].set_title("Matched rows by geometric distance rank")
    ax[1, 0].set_xlabel("Distance percentile within region")
    view = result.sort_values("historical_vol4_p1")
    ax[1, 1].bar(view.prototype.astype(str), view.historical_vol4_p1, color="#52799d")
    ax[1, 1].tick_params(axis="x", rotation=90, labelsize=6)
    ax[1, 1].set_title("K48 historical high-volatility probability by region")
    fig.tight_layout()
    fig.savefig(OUT / "07_stage_synthesis.png", dpi=160)
    geometry_scores = pd.read_csv(OUT / "geometry_all_scores.csv")
    geometry_view = geometry_scores.loc[geometry_scores.phase == "validation"].sort_values("logloss")
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(geometry_view.logloss, np.arange(len(geometry_view)), s=45)
    ax.set_yticks(np.arange(len(geometry_view)), geometry_view.model.astype(str))
    ax.set_xlim(geometry_view.logloss.min() - .001, geometry_view.logloss.max() + .001)
    ax.invert_yaxis()
    ax.set_title("Geometry-only candidates: development validation (zoomed)")
    ax.set_xlabel("Three-task log loss; lower is better")
    fig.tight_layout()
    fig.savefig(OUT / "02_geometry_candidates.png", dpi=160)
    conditional_scores = pd.read_csv(OUT / "conditional_all_scores.csv")
    conditional_view = conditional_scores.loc[conditional_scores.phase == "validation"].sort_values("logloss")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(conditional_view.logloss, np.arange(len(conditional_view)), s=65)
    ax.set_yticks(np.arange(len(conditional_view)), conditional_view.model.astype(str))
    ax.set_xlim(conditional_view.logloss.min() - .003, conditional_view.logloss.max() + .003)
    ax.invert_yaxis()
    ax.set_title("Background, conditional state, and direct models (zoomed)")
    ax.set_xlabel("Three-task log loss; lower is better")
    fig.tight_layout()
    fig.savefig(OUT / "03_conditional_comparison.png", dpi=160)
    print("saved", len(result), "prototype cards and", len(results), "paired contrasts", flush=True)


if __name__ == "__main__":
    main()
