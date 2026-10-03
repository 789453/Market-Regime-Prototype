"""Run the complete Stage 5 dual-formal-portfolio research workflow."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.backtest import aggregate_portfolio, run_event_backtest
from src.fusion import fuse_signals
from src.metrics import deflated_sharpe_ratio, performance_summary, stationary_block_bootstrap
from src.sizing import causal_daily_correlations, scale_portfolio_targets, size_instruments


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "reports" / "tables"
FIGS = ROOT / "reports" / "figs"
TRADABLE = ("ES", "NQ", "RTY", "HSI")
DETECTORS = tuple(f"D{i:02d}" for i in range(1, 9))
PORTFOLIOS = {"d01": ("D01",), "d01_d08": DETECTORS}
HOLDOUT = pd.Timestamp("2026-07-01", tz="UTC")
K_BASE = 168


def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict, dict]:
    bars = pd.read_parquet(ROOT / "data" / "interim" / "bars_5m.parquet").sort_index()
    context = pd.read_parquet(PROCESSED / "context.parquet").sort_index()
    labels = pd.read_parquet(PROCESSED / "labels.parquet").sort_index()
    signals = {
        detector: pd.read_parquet(PROCESSED / f"signals_stage4_{detector.lower()}.parquet").sort_index()
        for detector in DETECTORS
    }
    with open(ROOT / "configs" / "backtest.yaml", encoding="utf-8") as handle:
        backtest_config = yaml.safe_load(handle)
    with open(ROOT / "configs" / "cost_model.yaml", encoding="utf-8") as handle:
        cost_config = yaml.safe_load(handle)
    raw_cost = context["spread_bp"].copy()
    for root in context.index.get_level_values("root").unique():
        spec = cost_config["instruments"][root]
        mask = context.index.get_level_values("root") == root
        raw_cost.loc[mask] += (
            2 * spec["slippage_ticks_per_side"] * spec["tick_bp"]
            + 2 * spec["commission_bp_per_side"]
        )
    context["round_trip_cost_bp"] = raw_cost
    return bars, context, labels, signals, backtest_config, cost_config


def _strategy_metrics(backtest: pd.DataFrame, daily: pd.DataFrame, trials: int) -> dict[str, float]:
    result = performance_summary(daily["net_return"])
    result.update(deflated_sharpe_ratio(daily["net_return"], trials))
    gross = float(daily["gross_return"].sum())
    cost = float(daily["cost"].sum())
    executions = int(backtest["executed"].sum())
    turnover = float(backtest["turnover"].sum())
    ratios = backtest.loc[backtest["turnover"] > 0, "turnover"]
    result.update(
        {
            "gross_cumulative_log_return": gross,
            "total_cost": cost,
            "cost_to_gross_ratio": cost / gross if gross > 0 else np.inf,
            "annualized_turnover": float(daily["turnover"].mean() * 252),
            "executions": executions,
            "average_cost_per_turnover_bp": cost / turnover * 1e4 if turnover > 0 else np.nan,
            "cost_to_annual_volatility": cost / result["annualized_volatility"] if result["annualized_volatility"] > 0 else np.nan,
            "volume_limited_fraction": float(backtest["volume_limited"].mean()),
            "average_execution_delay_bars": float(backtest.loc[backtest.executed, "execution_delay_bars"].mean()),
            "active_bar_fraction": float((backtest["position"] != 0).mean()),
            "mean_trade_size": float(ratios.mean()) if len(ratios) else np.nan,
        }
    )
    return result


def _holding_periods(backtest: pd.DataFrame) -> pd.DataFrame:
    records = []
    for root, frame in backtest.groupby(level="root", sort=False):
        position = frame["position"].to_numpy(dtype=float)
        timestamps = frame.index.get_level_values("timestamp")
        start = None
        sign = 0.0
        for i, value in enumerate(position):
            new_sign = float(np.sign(value))
            if start is not None and (new_sign == 0 or new_sign != sign):
                records.append(
                    {
                        "root": root,
                        "side": "long" if sign > 0 else "short",
                        "bars": i - start,
                        "hours": (timestamps[i - 1] - timestamps[start]).total_seconds() / 3600 + 5 / 60,
                    }
                )
                start = None
            if start is None and new_sign != 0:
                start, sign = i, new_sign
        if start is not None:
            records.append(
                {
                    "root": root,
                    "side": "long" if sign > 0 else "short",
                    "bars": len(position) - start,
                    "hours": (timestamps[-1] - timestamps[start]).total_seconds() / 3600 + 5 / 60,
                }
            )
    return pd.DataFrame(records)


def _attributions(
    name: str,
    backtest: pd.DataFrame,
    daily: pd.DataFrame,
    raw_gate: pd.Series,
    leave_one_out: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    instrument = backtest.groupby(level="root")[["gross_return", "cost", "net_return"]].sum()
    instrument["share_of_total_net"] = instrument.net_return / backtest.net_return.sum()
    tables["instrument"] = instrument.reset_index()
    tables["detector"] = leave_one_out.copy()
    phase = backtest.groupby("session_phase")[["gross_return", "cost", "net_return"]].sum()
    phase["share_of_total_net"] = phase.net_return / backtest.net_return.sum()
    tables["phase"] = phase.reset_index()
    gate_frame = backtest[["gross_return", "cost", "net_return"]].copy()
    gate_frame["raw_gate_score"] = raw_gate.reindex(gate_frame.index)
    ranked = gate_frame["raw_gate_score"].rank(method="first", pct=True)
    gate_frame["gate_decile"] = np.minimum(np.floor(ranked * 10), 9).fillna(-1).astype(int) + 1
    tables["gate"] = gate_frame.groupby("gate_decile")[["gross_return", "cost", "net_return"]].sum().reset_index()
    month = daily.copy()
    month["month"] = month.index.to_period("M").astype(str)
    tables["month"] = month.groupby("month")[["gross_return", "cost", "net_return"]].sum().reset_index()
    side_frame = backtest.copy()
    side_frame["side"] = np.where(side_frame.position > 0, "long", np.where(side_frame.position < 0, "short", "flat"))
    side_rows = []
    for side in ("long", "short"):
        selected = side_frame[side_frame.side == side]
        date = selected.index.get_level_values("timestamp").date
        side_daily = selected.groupby(date)["net_return"].sum()
        row = {"side": side, **performance_summary(side_daily)}
        side_rows.append(row)
    tables["long_short"] = pd.DataFrame(side_rows)
    tables["holding"] = _holding_periods(backtest)
    for key, table in tables.items():
        table.to_csv(TABLES / f"05_{name}_attribution_{key}.csv", index=False)
    return tables


def _plot_portfolio(
    name: str,
    daily: pd.DataFrame,
    backtest: pd.DataFrame,
    attribution: dict[str, pd.DataFrame],
    sensitivity: pd.DataFrame,
    shuffle: pd.DataFrame,
    real_sharpe: float,
) -> None:
    gross_equity = np.exp(daily.gross_return.cumsum())
    net_equity = np.exp(daily.net_return.cumsum())
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gross_equity.index, gross_equity, "--", color="0.45", label="Gross")
    ax.plot(net_equity.index, net_equity, color="#16794a", label="Net")
    ax.set(title=f"Stage 5 {name}: Gross and net equity", ylabel="Growth of 1")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIGS / f"05_equity_{name}.png", dpi=150); plt.close(fig)

    equity = net_equity
    dd = equity / equity.cummax() - 1
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(dd.index, dd, 0, color="#b43c39", alpha=.65)
    ax.set(title=f"Stage 5 {name}: Drawdown", ylabel="Drawdown")
    fig.tight_layout(); fig.savefig(FIGS / f"05_drawdown_{name}.png", dpi=150); plt.close(fig)

    rolling = daily.net_return.rolling(60, min_periods=30).mean() / daily.net_return.rolling(60, min_periods=30).std() * np.sqrt(252)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rolling.index, rolling, color="#295f98"); ax.axhline(0, color="0.5", lw=.8)
    ax.set(title=f"Stage 5 {name}: Rolling 60-session Sharpe", ylabel="Sharpe")
    fig.tight_layout(); fig.savefig(FIGS / f"05_rolling_sharpe_{name}.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    specs = [("instrument", "root"), ("detector", "detector"), ("phase", "session_phase"),
             ("gate", "gate_decile"), ("month", "month"), ("long_short", "side")]
    for ax, (key, label) in zip(axes.flat, specs):
        table = attribution[key]
        if "net_return" in table:
            value = "net_return"
        elif "sharpe_delta" in table:
            value = "sharpe_delta"
        else:
            value = "sharpe"
        ax.bar(table[label].astype(str), table[value], color=np.where(table[value] >= 0, "#16794a", "#b43c39"))
        ax.set_title(key.replace("_", " ").title()); ax.tick_params(axis="x", rotation=45)
    fig.suptitle(f"Stage 5 {name}: Required attribution views")
    fig.tight_layout(); fig.savefig(FIGS / f"05_attribution_{name}.png", dpi=150); plt.close(fig)

    holding = attribution["holding"]
    fig, ax = plt.subplots(figsize=(9, 4))
    for root, frame in holding.groupby("root"):
        ax.hist(frame.hours.clip(upper=frame.hours.quantile(.99)), bins=30, alpha=.45, label=root)
    ax.set(title=f"Stage 5 {name}: Holding period distribution", xlabel="Hours", ylabel="Bets")
    ax.legend(); fig.tight_layout(); fig.savefig(FIGS / f"05_holding_{name}.png", dpi=150); plt.close(fig)

    series = backtest.groupby(backtest.index.get_level_values("timestamp").date)[["turnover", "cost"]].sum()
    fig, ax = plt.subplots(figsize=(10, 4)); ax.plot(series.index, series.turnover, label="Turnover", color="#295f98")
    ax2 = ax.twinx(); ax2.plot(series.index, series.cost * 1e4, label="Cost", color="#b43c39", alpha=.65)
    ax.set(title=f"Stage 5 {name}: Turnover and cost", ylabel="Turnover"); ax2.set_ylabel("Cost (bp)")
    fig.tight_layout(); fig.savefig(FIGS / f"05_turnover_cost_{name}.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, dimension in zip(axes, ("cost_multiplier", "delay", "band_zeta")):
        frame = sensitivity[sensitivity.dimension == dimension]
        ax.plot(frame.setting.astype(str), frame.sharpe, marker="o"); ax.set(title=dimension, ylabel="Sharpe")
    fig.suptitle(f"Stage 5 {name}: Core sensitivity panel")
    fig.tight_layout(); fig.savefig(FIGS / f"05_sensitivity_{name}.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4)); ax.hist(shuffle.sharpe, bins=30, color="0.55", alpha=.8)
    ax.axvline(real_sharpe, color="#b43c39", lw=2, label=f"Real={real_sharpe:.2f}")
    ax.set(title=f"Stage 5 {name}: 500 circular-shift null", xlabel="Sharpe"); ax.legend()
    fig.tight_layout(); fig.savefig(FIGS / f"05_shuffle_{name}.png", dpi=150); plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True); FIGS.mkdir(parents=True, exist_ok=True)
    bars, context, labels, signals, config, _ = _load()
    tradable_idx = bars.index[bars.index.get_level_values("root").isin(TRADABLE)]
    bars = bars.reindex(tradable_idx); context = context.reindex(tradable_idx); labels = labels.reindex(tradable_idx)
    signals = {key: value.reindex(tradable_idx) for key, value in signals.items()}
    correlations = causal_daily_correlations(bars, context, roots=TRADABLE)
    fusion_cache: dict[tuple[str, ...], tuple[pd.DataFrame, pd.DataFrame]] = {}
    strategy_cache: dict[tuple, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}

    def build(
        detector_set: tuple[str, ...], *, cost_multiplier: float = 1.5,
        execution_price: str = "wap", band_zeta: float = .05,
        vol_target: float = .10, extra_delay: int = 0,
        roots: tuple[str, ...] = TRADABLE, mu_override: pd.Series | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        key = (detector_set, cost_multiplier, execution_price, band_zeta, vol_target, extra_delay, roots,
               None if mu_override is None else id(mu_override))
        if key in strategy_cache:
            return strategy_cache[key]
        if detector_set not in fusion_cache:
            fusion_cache[detector_set] = fuse_signals({d: signals[d] for d in detector_set}, tau=.5)
        fused = fusion_cache[detector_set][0].copy()
        if mu_override is not None:
            fused["mu"] = mu_override.reindex(fused.index).fillna(0.0)
        sized = size_instruments(
            fused, context, instrument_vol_target=vol_target, portfolio_vol_target=vol_target,
            band_zeta=band_zeta,
        )
        sized = scale_portfolio_targets(
            sized, correlations, roots=TRADABLE, portfolio_vol_target=vol_target
        )
        selected_idx = bars.index[bars.index.get_level_values("root").isin(roots)]
        decisions = sized.reindex(selected_idx).copy()
        decisions["round_trip_cost_bp"] = context.reindex(selected_idx)["round_trip_cost_bp"]
        bt = run_event_backtest(
            bars.reindex(selected_idx), decisions, execution_price=execution_price,
            cost_multiplier=cost_multiplier, delay_bars=1 + extra_delay,
            avoid_session_edge_bars=config["execution"]["avoid_session_edge_bars"],
            avoid_roll_boundary_bars=config["execution"]["avoid_roll_boundary_bars"],
            max_volume_participation=config["execution"]["max_volume_participation"],
        )
        intraday, daily = aggregate_portfolio(bt)
        strategy_cache[key] = (fused, sized, bt, daily)
        return strategy_cache[key]

    baseline: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    baseline_metrics = []
    for name, detector_set in PORTFOLIOS.items():
        baseline[name] = build(detector_set)
        fused, sized, bt, daily = baseline[name]
        metrics = _strategy_metrics(bt, daily, K_BASE)
        holdout_daily = daily.loc[daily.index >= HOLDOUT.tz_localize(None)]
        holdout = performance_summary(holdout_daily.net_return)
        metrics.update({f"holdout_{key}": value for key, value in holdout.items()})
        metrics["portfolio"] = name; metrics["detectors"] = ",".join(detector_set)
        baseline_metrics.append(metrics)
        fused.to_parquet(PROCESSED / f"fused_stage5_{name}.parquet")
        sized.to_parquet(PROCESSED / f"sizing_stage5_{name}.parquet")
        bt.to_parquet(PROCESSED / f"backtest_stage5_{name}.parquet")
        daily.to_csv(TABLES / f"05_{name}_daily_returns.csv")
    baseline_table = pd.DataFrame(baseline_metrics).set_index("portfolio")

    single_rows = []
    for detector in DETECTORS:
        _, _, bt, daily = build((detector,))
        row = _strategy_metrics(bt, daily, K_BASE); row["detector"] = detector
        single_rows.append(row)
    single_table = pd.DataFrame(single_rows)
    single_table.to_csv(TABLES / "05_single_detector_backtests.csv", index=False)

    sensitivities: dict[str, pd.DataFrame] = {}
    leaveouts: dict[str, pd.DataFrame] = {}
    for name, detector_set in PORTFOLIOS.items():
        rows = []
        dimensions = {
            "cost_multiplier": [(str(x), {"cost_multiplier": x}) for x in (1.0, 1.5, 2.0, 3.0)],
            "execution_price": [(x, {"execution_price": x}) for x in ("open", "wap", "mid")],
            "band_zeta": [(str(x), {"band_zeta": x}) for x in (.025, .05, .075)],
            "vol_target": [(str(x), {"vol_target": x}) for x in (.05, .10, .15)],
            "delay": [(str(x), {"extra_delay": x}) for x in (0, 1, 2, 3)],
            "root_subset": [("ES", {"roots": ("ES",)}), ("ES_NQ_RTY", {"roots": ("ES", "NQ", "RTY")}), ("ALL", {"roots": TRADABLE})],
        }
        for dimension, settings in dimensions.items():
            for setting, kwargs in settings:
                _, _, bt, daily = build(detector_set, **kwargs)
                metric = _strategy_metrics(bt, daily, K_BASE)
                rows.append({"dimension": dimension, "setting": setting, **metric})
        loo_rows = []
        for detector in detector_set:
            reduced = tuple(d for d in detector_set if d != detector)
            if reduced:
                _, _, bt, daily = build(reduced)
                metric = _strategy_metrics(bt, daily, K_BASE)
            else:
                metric = {"sharpe": 0.0, "annualized_return": 0.0, "net_return": 0.0}
            loo_rows.append(
                {"detector": detector, "sharpe_without": metric.get("sharpe", np.nan),
                 "sharpe_delta": baseline_table.loc[name, "sharpe"] - metric.get("sharpe", np.nan),
                 "annualized_return_without": metric.get("annualized_return", np.nan)}
            )
            rows.append({"dimension": "detector_leave_one_out", "setting": detector, **metric})
        leaveouts[name] = pd.DataFrame(loo_rows)
        sensitivities[name] = pd.DataFrame(rows)
        sensitivities[name].to_csv(TABLES / f"05_{name}_sensitivity.csv", index=False)

    # Injection audits use the full formal portfolio and the identical sizing/backtest path.
    all_fused = baseline["d01_d08"][0]
    fwd = labels["fwd_ret_z_6"].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-8, 8)
    injected_mu = .95 * all_fused.mu + .05 * fwd
    _, _, inj_bt, inj_daily = build(DETECTORS, mu_override=injected_mu)
    future_mu = pd.Series(index=all_fused.index, dtype=float)
    for _, loc in all_fused.groupby(level="root", sort=False).groups.items():
        values = all_fused.loc[loc, "mu"].to_numpy()
        future_mu.loc[loc] = np.r_[values[1:], 0.0]
    _, _, lead_bt, lead_daily = build(DETECTORS, mu_override=future_mu)
    rng = np.random.default_rng(20260905)
    noise = pd.Series(0.0, index=all_fused.index)
    active_rate = float((all_fused.mu != 0).mean())
    mask = rng.random(len(noise)) < active_rate
    noise.iloc[np.flatnonzero(mask)] = rng.normal(0, all_fused.loc[all_fused.mu != 0, "mu"].std(), mask.sum())
    _, _, noise_bt, noise_daily = build(DETECTORS, mu_override=noise)
    audit_rows = [
        {"test": "baseline", **_strategy_metrics(baseline["d01_d08"][2], baseline["d01_d08"][3], K_BASE)},
        {"test": "positive_5pct_future_return", **_strategy_metrics(inj_bt, inj_daily, K_BASE)},
        {"test": "future_signal_one_bar", **_strategy_metrics(lead_bt, lead_daily, K_BASE)},
        {"test": "pure_random_signal", **_strategy_metrics(noise_bt, noise_daily, K_BASE)},
    ]
    audit = pd.DataFrame(audit_rows)
    audit["gross_sharpe"] = [
        performance_summary(frame["gross_return"])["sharpe"]
        for frame in (baseline["d01_d08"][3], inj_daily, lead_daily, noise_daily)
    ]

    # Prefix causality for newly implemented modules.
    cutoff_time = bars.index.get_level_values("timestamp").sort_values()[int(len(bars) * .60)]
    prefix_idx = bars.index[bars.index.get_level_values("timestamp") <= cutoff_time]
    prefix_signals = {d: signals[d].reindex(prefix_idx) for d in DETECTORS}
    prefix_fused, _ = fuse_signals(prefix_signals, tau=.5)
    fusion_diff = float((prefix_fused.mu - all_fused.reindex(prefix_idx).mu).abs().max())
    prefix_context = context.reindex(prefix_idx); prefix_bars = bars.reindex(prefix_idx)
    prefix_sized = size_instruments(prefix_fused, prefix_context)
    prefix_corr = causal_daily_correlations(prefix_bars, prefix_context, roots=TRADABLE)
    prefix_sized = scale_portfolio_targets(prefix_sized, prefix_corr, roots=TRADABLE)
    sizing_diff = float((prefix_sized.target - baseline["d01_d08"][1].reindex(prefix_idx).target).abs().max())
    audit["fusion_prefix_max_abs_diff"] = fusion_diff
    audit["sizing_prefix_max_abs_diff"] = sizing_diff
    audit.to_csv(TABLES / "05_audit_injections_and_causality.csv", index=False)

    # Full 500-draw circular-shift null for both formal portfolios.
    shuffle_tables: dict[str, pd.DataFrame] = {}
    for portfolio_number, (name, detector_set) in enumerate(PORTFOLIOS.items()):
        _, sized, _, _ = baseline[name]
        shuffle_path = TABLES / f"05_{name}_shuffle_500_exitfix.csv"
        if shuffle_path.exists() and len(pd.read_csv(shuffle_path)) == config["validation"]["signal_shuffle_draws"]:
            shuffle = pd.read_csv(shuffle_path)[["draw", "sharpe"]]
        else:
            draws = []
            rng = np.random.default_rng(20260905 + portfolio_number)
            for draw in range(config["validation"]["signal_shuffle_draws"]):
                shifted = sized.copy()
                for _, loc in shifted.groupby(level="root", sort=False).groups.items():
                    values = shifted.loc[loc, "target"].to_numpy()
                    offset = int(rng.integers(1, len(values)))
                    shifted.loc[loc, "target"] = np.roll(values, offset)
                shifted["round_trip_cost_bp"] = context["round_trip_cost_bp"]
                bt = run_event_backtest(
                    bars, shifted, execution_price="wap", cost_multiplier=1.5, delay_bars=1,
                    avoid_session_edge_bars=2, avoid_roll_boundary_bars=3, max_volume_participation=.05,
                )
                _, daily = aggregate_portfolio(bt)
                draws.append({"draw": draw, "sharpe": performance_summary(daily.net_return)["sharpe"]})
            shuffle = pd.DataFrame(draws)
        real = baseline_table.loc[name, "sharpe"]
        percentile = float((shuffle.sharpe <= real).mean())
        shuffle["real_sharpe"] = real; shuffle["real_percentile"] = percentile
        shuffle.to_csv(shuffle_path, index=False)
        shuffle_tables[name] = shuffle
        baseline_table.loc[name, "shuffle_percentile"] = percentile

    attributions = {}
    for name, detector_set in PORTFOLIOS.items():
        fused, _, bt, daily = baseline[name]
        raw_gate = pd.concat([signals[d]["raw_gate_score"] for d in detector_set], axis=1).mean(axis=1)
        attributions[name] = _attributions(name, bt, daily, raw_gate, leaveouts[name])
        bootstrap = stationary_block_bootstrap(daily.net_return, draws=1000)
        bootstrap.to_csv(TABLES / f"05_{name}_bootstrap.csv", index=False)
        baseline_table.loc[name, "bootstrap_sharpe_p025"] = bootstrap.sharpe.quantile(.025)
        baseline_table.loc[name, "bootstrap_sharpe_p975"] = bootstrap.sharpe.quantile(.975)
        holding = attributions[name]["holding"]
        baseline_table.loc[name, "average_holding_hours"] = holding.hours.mean() if len(holding) else np.nan
        _plot_portfolio(name, daily, bt, attributions[name], sensitivities[name], shuffle_tables[name], baseline_table.loc[name, "sharpe"])

    best_single = float(single_table.sharpe.max())
    baseline_table["best_single_detector_sharpe"] = best_single
    baseline_table["g5_diversification_pass"] = baseline_table.sharpe > 1.15 * best_single
    baseline_table["g5_cost_pass"] = baseline_table.cost_to_gross_ratio < .35
    baseline_table["g5_shuffle_pass"] = baseline_table.shuffle_percentile >= .95
    injection_pass = (
        audit.loc[audit.test == "positive_5pct_future_return", "sharpe"].iloc[0] > 5
        and audit.loc[audit.test == "future_signal_one_bar", "sharpe"].iloc[0] > audit.loc[audit.test == "baseline", "sharpe"].iloc[0]
        and abs(audit.loc[audit.test == "pure_random_signal", "gross_sharpe"].iloc[0]) < .5
        and audit.loc[audit.test == "pure_random_signal", "cumulative_return"].iloc[0] < 0
        and audit.loc[audit.test == "pure_random_signal", "total_cost"].iloc[0] > 0
    )
    baseline_table["g5_injection_pass"] = injection_pass
    baseline_table["g5_all_pass"] = baseline_table[["g5_diversification_pass", "g5_cost_pass", "g5_shuffle_pass", "g5_injection_pass"]].all(axis=1)
    baseline_table.to_csv(TABLES / "05_portfolio_metrics.csv")
    manifest = {
        "stage": 5, "formal_portfolios": {k: list(v) for k, v in PORTFOLIOS.items()},
        "tradable_roots": list(TRADABLE), "context_only_roots": ["HTI"],
        "cost_multiplier": 1.5, "holdout_start": str(HOLDOUT), "holdout_views_used": 1,
        "shuffle_draws_per_portfolio": 500, "global_k_before_stage5": K_BASE,
    }
    (PROCESSED / "stage5_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(baseline_table.to_string())
    print("AUDIT")
    print(audit[["test", "sharpe", "annualized_return", "total_cost"]].to_string(index=False))


if __name__ == "__main__":
    main()
