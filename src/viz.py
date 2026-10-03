"""Project-wide diagnostic visualization helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.stattools import acf


ROOT_ORDER = ["ES", "NQ", "RTY", "HSI", "HTI"]


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_local_hour_month_heatmaps(minute: pd.DataFrame, output_dir: Path) -> list[Path]:
    paths = []
    work = minute.loc[~minute["is_filled"]].copy()
    work["month"] = pd.to_datetime(work["session_id"]).dt.strftime("%Y-%m")
    work["local_hour"] = work["local_minute"] // 60
    for root in ROOT_ORDER:
        table = (
            work.loc[work["root"].eq(root)]
            .groupby(["month", "local_hour"], observed=True)
            .size()
            .unstack(fill_value=0)
            .reindex(columns=range(24), fill_value=0)
        )
        fig, ax = plt.subplots(figsize=(11, 3.8))
        image = ax.imshow(table.to_numpy(), aspect="auto", cmap="viridis")
        ax.set_yticks(range(len(table.index)), table.index)
        ax.set_xticks(range(24), range(24))
        ax.set_xlabel("Exchange-local hour")
        ax.set_ylabel("Session month")
        ax.set_title(f"Stage 1 · {root} valid 1-minute bar count by local hour")
        fig.colorbar(image, ax=ax, label="Valid bar count")
        path = output_dir / f"01_local_hour_valid_{root}.png"
        _save(fig, path)
        paths.append(path)
    return paths


def plot_filled_heatmap(minute: pd.DataFrame, output_dir: Path) -> Path:
    work = minute.copy()
    work["local_hour"] = work["local_minute"] // 60
    table = (
        work.groupby(["root", "local_hour"], observed=True)["is_filled"]
        .mean()
        .unstack()
        .reindex(index=ROOT_ORDER, columns=range(24))
    )
    fig, ax = plt.subplots(figsize=(11, 3.8))
    image = ax.imshow(table.to_numpy(), aspect="auto", cmap="magma", vmin=0, vmax=max(0.1, np.nanmax(table.to_numpy())))
    ax.set_yticks(range(len(table.index)), table.index)
    ax.set_xticks(range(24), range(24))
    ax.set_xlabel("Exchange-local hour")
    ax.set_ylabel("Root")
    ax.set_title("Stage 1 · Filled 1-minute bar ratio by root and local hour")
    fig.colorbar(image, ax=ax, label="Filled ratio")
    path = output_dir / "01_filled_ratio_heatmap.png"
    _save(fig, path)
    return path


def plot_session_endpoints(daily: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(len(ROOT_ORDER), 1, figsize=(12, 10), sharex=True)
    for ax, root in zip(axes, ROOT_ORDER):
        part = daily.loc[daily["root"].eq(root)]
        ax.plot(pd.to_datetime(part["session_id"]), part["start_minute"] / 60, ".", ms=2.5, label="start")
        ax.plot(pd.to_datetime(part["session_id"]), part["end_minute"] / 60, ".", ms=2.5, label="end")
        ax.set_ylabel(root)
        ax.grid(alpha=0.2)
    axes[0].legend(loc="upper right", ncol=2)
    axes[-1].set_xlabel("Session id")
    fig.suptitle("Stage 1 · Observed exchange-local session endpoints")
    path = output_dir / "01_session_endpoints.png"
    _save(fig, path)
    return path


def plot_roll_jumps(rolls: pd.DataFrame, output_dir: Path) -> Path:
    labels = rolls["root"] + " " + rolls["previous_symbol"] + "→" + rolls["next_symbol"]
    colors = np.where(rolls["jump_bp"] >= 0, "#2ca02c", "#d62728")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(rolls)), rolls["jump_bp"], color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(rolls)), labels, rotation=70, ha="right")
    ax.set_ylabel("Log price jump (bp)")
    ax.set_title("Stage 1 · Contract roll boundary price jumps")
    path = output_dir / "01_roll_boundary_jumps.png"
    _save(fig, path)
    return path


def plot_spread_curves(curve: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5))
    for root in ROOT_ORDER:
        part = curve.loc[curve["root"].eq(root)].sort_values("local_hour")
        ax.plot(part["local_hour"], part["spread_bp"], marker="o", ms=3, label=root)
    ax.set_yscale("log")
    ax.set_xticks(range(24))
    ax.set_xlabel("Exchange-local hour")
    ax.set_ylabel("Effective spread (bp, log scale)")
    ax.set_title("Stage 1 · CHL effective spread with one-tick floor")
    ax.grid(alpha=0.2)
    ax.legend()
    path = output_dir / "01_effective_spread.png"
    _save(fig, path)
    return path


def plot_activity_curves(minute: pd.DataFrame, output_dir: Path) -> Path:
    work = minute.loc[~minute["is_filled"]].copy()
    activity = work.groupby(["root", "local_minute"], observed=True)[["volume", "bar_count"]].median().reset_index()
    fig, axes = plt.subplots(len(ROOT_ORDER), 2, figsize=(13, 12), sharex=False)
    for row, root in enumerate(ROOT_ORDER):
        part = activity.loc[activity["root"].eq(root)]
        axes[row, 0].plot(part["local_minute"] / 60, part["volume"], color="#1f77b4", lw=0.8)
        axes[row, 1].plot(part["local_minute"] / 60, part["bar_count"], color="#9467bd", lw=0.8)
        axes[row, 0].set_ylabel(root)
        axes[row, 0].grid(alpha=0.15)
        axes[row, 1].grid(alpha=0.15)
    axes[0, 0].set_title("Median volume")
    axes[0, 1].set_title("Median trade count")
    axes[-1, 0].set_xlabel("Exchange-local hour")
    axes[-1, 1].set_xlabel("Exchange-local hour")
    fig.suptitle("Stage 1 · Intraday activity curves")
    path = output_dir / "01_intraday_activity.png"
    _save(fig, path)
    return path


def plot_lead_lag(correlations: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(correlations["lag"], correlations["correlation"], marker="o", color="#1f77b4")
    best = correlations.loc[correlations["correlation"].idxmax()]
    ax.axvline(best["lag"], color="#d62728", ls="--", label=f"peak lag={int(best['lag'])}")
    ax.set_xticks(correlations["lag"])
    ax.set_xlabel("k in Corr(ES[t-k], HSI[t]) (5-minute bars)")
    ax.set_ylabel("Correlation")
    ax.set_title("Stage 1 · ES–HSI lead-lag correlation")
    ax.grid(alpha=0.2)
    ax.legend()
    path = output_dir / "01_es_hsi_lead_lag.png"
    _save(fig, path)
    return path


def plot_seasonality_before_after(vol_bars: pd.DataFrame, output_dir: Path) -> list[Path]:
    paths = []
    work = vol_bars.loc[vol_bars["seasonality_ready"]].copy()
    work["before"] = (work["ret"] / work["sigma_day"]).abs()
    work["after"] = (work["ret"] / work["sigma_hat"]).abs()
    for root in ROOT_ORDER:
        part = work.loc[work["root"].eq(root)]
        curve = part.groupby("local_minute")[["before", "after"]].mean().dropna()
        elapsed = (curve.index.to_numpy() - (18 * 60 if root in {"ES", "NQ", "RTY"} else 17 * 60)) % (24 * 60)
        order = np.argsort(elapsed)
        x = elapsed[order] / 60
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for ax, column, title in zip(axes, ["before", "after"], ["Before seasonal removal", "After full normalization"]):
            values = curve[column].to_numpy()[order]
            ax.plot(x, values, color="#b0b0b0", alpha=0.5, lw=0.7)
            ax.plot(x, lowess(values, x, frac=0.10, return_sorted=False), color="#1f77b4", lw=2)
            ax.set_title(title)
            ax.set_xlabel("Hours since session open anchor")
            ax.grid(alpha=0.2)
        axes[0].set_ylabel("Mean absolute standardized return")
        fig.suptitle(f"Stage 2 · {root} intraday seasonality normalization", y=0.99)
        fig.subplots_adjust(top=0.78, wspace=0.20)
        path = output_dir / f"02_seasonality_{root}.png"
        _save(fig, path)
        paths.append(path)
    return paths


def plot_har_diagnostics(daily: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(2, len(ROOT_ORDER), figsize=(16, 7))
    for column, root in enumerate(ROOT_ORDER):
        history = daily.loc[daily["root"].eq(root)].sort_values("session_id").copy()
        history["actual_log_var"] = np.log(history["continuous_var"])
        history["historical_mean"] = history["actual_log_var"].expanding().mean().shift(1)
        group = history.dropna(subset=["har_log_var", "actual_log_var", "historical_mean"])
        actual = group["actual_log_var"]
        axes[0, column].scatter(group["har_log_var"], actual, s=10, alpha=0.55)
        if len(group):
            bounds = [min(actual.min(), group["har_log_var"].min()), max(actual.max(), group["har_log_var"].max())]
            axes[0, column].plot(bounds, bounds, ls="--", color="black", lw=0.8)
        axes[0, column].set_title(root)
        axes[0, column].set_xlabel("Predicted log variance")
        squared = np.square(actual - group["har_log_var"])
        benchmark_squared = np.square(actual - group["historical_mean"])
        rolling_r2 = 1 - squared.rolling(20, min_periods=10).sum() / benchmark_squared.rolling(20, min_periods=10).sum()
        axes[1, column].plot(pd.to_datetime(group["session_id"]), rolling_r2, color="#1f77b4")
        axes[1, column].axhline(0.4, color="#d62728", ls="--", lw=0.8)
        axes[1, column].set_xlabel("Session")
        axes[1, column].tick_params(axis="x", rotation=45)
    axes[0, 0].set_ylabel("Realized log variance")
    axes[1, 0].set_ylabel("Rolling OOS R² (20 sessions)")
    fig.suptitle("Stage 2 · Expanding constrained HAR-J diagnostics")
    path = output_dir / "02_har_diagnostics.png"
    _save(fig, path)
    return path


def plot_sigma_calibration(vol_bars: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(2, len(ROOT_ORDER), figsize=(17, 7.2))
    for column, root in enumerate(ROOT_ORDER):
        part = vol_bars.loc[vol_bars["root"].eq(root)].dropna(subset=["ret", "sigma_hat"]).copy()
        rolling_predicted = part["sigma_hat"].rolling(120, min_periods=60).median()
        rolling_realized = part["ret"].abs().rolling(120, min_periods=60).mean()
        axes[0, column].plot(part["timestamp"], rolling_predicted, lw=0.8, label="predicted median")
        axes[0, column].plot(part["timestamp"], rolling_realized, lw=0.8, label="realized mean |r|")
        axes[0, column].set_title(root)
        axes[0, column].tick_params(axis="x", rotation=35)
        axes[0, column].grid(alpha=0.2)
        part["bin"] = pd.qcut(part["sigma_hat"], 5, labels=False, duplicates="drop")
        calibration = part.groupby("bin").agg(predicted=("sigma_hat", "mean"), realized=("ret", lambda x: x.abs().mean()))
        axes[1, column].plot(calibration["predicted"], calibration["realized"], marker="o")
        axes[1, column].set_xlabel("Predicted sigma")
        axes[1, column].grid(alpha=0.2)
    axes[0, 0].set_ylabel("Rolling scale")
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].set_ylabel("Mean |return|")
    fig.suptitle("Stage 2 · Rolling sigma and conditional quantile calibration")
    path = output_dir / "02_sigma_calibration.png"
    _save(fig, path)
    return path


def plot_qq_acf(vol_bars: pd.DataFrame, output_dir: Path) -> list[Path]:
    paths = []
    for root in ROOT_ORDER:
        z = (vol_bars.loc[vol_bars["root"].eq(root), "ret"] / vol_bars.loc[vol_bars["root"].eq(root), "sigma_hat"]).dropna()
        z = z.clip(z.quantile(0.001), z.quantile(0.999))
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        stats.probplot(z, dist="norm", plot=axes[0])
        axes[0].set_title(f"{root} standardized-return QQ")
        correlations = acf(z, nlags=24, fft=True, missing="drop")
        axes[1].bar(range(1, len(correlations)), correlations[1:], color="#1f77b4")
        axes[1].axhline(0, color="black", lw=0.7)
        axes[1].set_xlabel("Lag (5-minute bars)")
        axes[1].set_ylabel("ACF")
        axes[1].set_title(f"{root} standardized-return ACF")
        path = output_dir / f"02_qq_acf_{root}.png"
        _save(fig, path)
        paths.append(path)
    return paths


def plot_context_panel(bars: pd.DataFrame, context: pd.DataFrame, output_dir: Path) -> Path:
    merged = bars[["root", "timestamp", "close"]].merge(context, on=["root", "timestamp"])
    part = merged.loc[merged["root"].eq("ES")].tail(2500)
    fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(part["timestamp"], part["close"], color="black")
    axes[0].set_ylabel("ES price")
    for ax, column in zip(axes[1:], ["sigma_pct", "vol_ratio", "er_20", "liq_z"]):
        ax.plot(part["timestamp"], part[column], lw=0.8)
        ax.set_ylabel(column)
        ax.grid(alpha=0.15)
    axes[-1].set_xlabel("UTC timestamp")
    fig.suptitle("Stage 2 · ES causal context panel (latest 2,500 bars)")
    path = output_dir / "02_context_panel_ES.png"
    _save(fig, path)
    return path


def plot_detector_horizons(cards: pd.DataFrame, output_dir: Path) -> Path:
    detectors = sorted(cards["detector_id"].unique())
    fig, axes = plt.subplots(4, 2, figsize=(12, 13), sharex=True)
    for ax, detector_id in zip(axes.ravel(), detectors):
        group = cards.loc[cards["detector_id"].eq(detector_id)].sort_values("horizon")
        ax.plot(group["horizon"], group["mean_z"], marker="o", label="gross standardized mean")
        ax.fill_between(group["horizon"], group["ci_low"], group["ci_high"], alpha=0.18)
        ax.axhline(0, color="black", lw=0.7)
        ax.set_title(detector_id)
        ax.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("Horizon (5-minute bars)")
    axes[-1, 1].set_xlabel("Horizon (5-minute bars)")
    axes[0, 0].set_ylabel("Signed fwd_ret_z")
    fig.suptitle("Stage 3 · Detector information-decay curves")
    path = output_dir / "03_detector_horizon_curves.png"
    _save(fig, path)
    return path


def plot_parameter_surface(surface_1d: pd.DataFrame, surface_2d: pd.DataFrame, detector_id: str, output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for parameter, group in surface_1d.groupby("parameter", sort=False):
        axes[0].plot(group["value"], group["performance"], marker="o", label=parameter)
    axes[0].axhline(0, color="black", lw=0.7)
    axes[0].set_xlabel("Parameter value")
    axes[0].set_ylabel("Signed mean at default horizon")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)
    if len(surface_2d):
        table = surface_2d.pivot(index="value_1", columns="value_2", values="performance")
        image = axes[1].imshow(table.to_numpy(), aspect="auto", cmap="RdBu_r")
        axes[1].set_xticks(range(len(table.columns)), table.columns)
        axes[1].set_yticks(range(len(table.index)), table.index)
        axes[1].set_xlabel(str(surface_2d["parameter_2"].iloc[0]))
        axes[1].set_ylabel(str(surface_2d["parameter_1"].iloc[0]))
        fig.colorbar(image, ax=axes[1], label="Signed mean")
    fig.suptitle(f"Stage 3 · {detector_id} registered parameter surface")
    path = output_dir / f"03_parameter_surface_{detector_id}.png"
    _save(fig, path)
    return path


def plot_null_distributions(nulls: dict[str, np.ndarray], actual: dict[str, float], output_dir: Path) -> Path:
    fig, axes = plt.subplots(4, 2, figsize=(12, 13))
    for ax, detector_id in zip(axes.ravel(), sorted(nulls)):
        values = nulls[detector_id]
        ax.hist(values[np.isfinite(values)], bins=40, color="#8da0cb", alpha=0.8)
        ax.axvline(actual[detector_id], color="#d62728", lw=1.5, label="actual")
        ax.set_title(detector_id)
        ax.legend()
    fig.suptitle("Stage 3 · Phase- and cluster-matched timing nulls")
    path = output_dir / "03_matched_nulls.png"
    _save(fig, path)
    return path


def plot_gate_curves(curves: dict[str, pd.DataFrame], output_dir: Path) -> Path:
    fig, axes = plt.subplots(4, 2, figsize=(12, 13))
    for ax, detector_id in zip(axes.ravel(), sorted(curves)):
        curve = curves[detector_id]
        if len(curve):
            ax.plot(curve["gate_mean"], curve["outcome_mean"], marker="o", color="#9aa0a6", label="raw decile mean")
            ax.plot(curve["gate_mean"], curve["isotonic"], marker="o", color="#1f77b4", label="isotonic")
        ax.axhline(0, color="black", lw=0.7)
        ax.set_title(detector_id)
        ax.set_xlabel("Detector-specific gate")
        ax.set_ylabel("Signed fwd_ret_z")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Stage 4 · Conditional effect by detector-specific gate decile")
    fig.subplots_adjust(hspace=0.48, wspace=0.28, top=0.94)
    path = output_dir / "04_gate_curves.png"
    _save(fig, path)
    return path


def plot_shrinkage_summary(summary: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(summary["detector_id"], summary["positive_tau_terms"], color="#4c78a8")
    axes[0].set_ylabel("Terms with tau² > 0")
    axes[0].set_title("Detected state heterogeneity")
    axes[1].bar(summary["detector_id"], summary["median_trigger_confidence"], color="#f58518")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Median causal posterior confidence")
    axes[1].set_title("Compound-cell confidence")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Stage 4 · Empirical-Bayes shrinkage diagnostics")
    path = output_dir / "04_shrinkage_summary.png"
    _save(fig, path)
    return path
