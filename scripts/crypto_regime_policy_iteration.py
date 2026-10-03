"""Fixed-forecast policy ablation: review cadence, switching cost and calm heads.

All forecasters and thresholds were produced by crypto_regime_prototype_stage.
This script changes only the action mapping, on the same exploratory history.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.crypto.regime_prototypes import REGIMES, adaptive_pair_weights
from src.crypto.signal_chain import ledger
SOURCE = ROOT / "reports/crypto/regime_prototypes_v1"
OUT = ROOT / "reports/crypto/regime_policy_iteration"
CFG = yaml.safe_load((ROOT / "configs/crypto_regime_prototypes.yaml").read_text(encoding="utf-8"))
RAW = Path(CFG["source"])
SYMBOLS = CFG["symbols"]
# name, forecast, field, continuous_size, threshold_mode, review_hours,
# switch_hurdle, target-weight deadband.  The first three isolate threshold and
# size at the same original hourly cadence; the others isolate slower action.
CASES = (("direct_uniform_continuous_hourly", "direct", "direct_alpha24", True, "uniform", 1, .0004, 0.),
         ("direct_state_fixed_hourly", "direct", "direct_alpha24", False, "state", 1, .0004, 0.),
         ("direct_state_continuous_hourly", "direct", "direct_alpha24", True, "state", 1, .0004, 0.),
         ("direct_sticky_continuous", "direct", "direct_alpha24", True, "state", 4, .0016, .10),
         ("direct_sticky_fixed", "direct", "direct_alpha24", False, "state", 4, .0016, .10),
         ("calm_specialist_sticky", "calm_specialist", "calm_alpha24", True, "state", 4, .0016, .10),
         ("directional_sticky", "directional", "direction_alpha24", True, "state", 4, .0016, .10),
         ("direction_hybrid_sticky", "direction_hybrid", "direction_hybrid_alpha24", True, "state", 4, .0016, .10))


def raw_entry_prices(times: pd.DatetimeIndex) -> np.ndarray:
    ms = (times.as_unit("ms").asi8 + 300_000).astype(np.int64)
    request = np.r_[ms, ms[-1] + 3_600_000]
    out = np.zeros((len(request), len(SYMBOLS)), dtype=np.float64)
    for j, symbol in enumerate(SYMBOLS):
        table = pq.read_table(RAW / symbol / "5m.parquet", columns=["open_time", "open"],
            filters=[("open_time", ">=", int(ms[0])), ("open_time", "<=", int(request[-1]))]).to_pandas()
        stamp = table.open_time.to_numpy(dtype=np.int64)
        hit = np.searchsorted(stamp, request)
        if np.any(hit >= len(stamp)) or not np.array_equal(stamp[hit], request):
            raise AssertionError(f"missing 5m execution open for {symbol}")
        out[:, j] = np.log(table.open.to_numpy(dtype=np.float64)[hit])
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = pd.read_parquet(SOURCE / "forward_forecasts.parquet")
    thresholds = pd.read_csv(SOURCE / "threshold_history.csv")
    card_map = {(row.month, row.forecast, row.regime, row.policy): (row.enter, row.exit)
                for row in thresholds.itertuples(index=False)}
    scores, attribution, hourly_rows = [], [], []
    for fold in sorted(rows.fold.unique()):
        if fold == 0:
            continue
        frame = rows[rows.fold == fold].copy()
        times = pd.DatetimeIndex(frame.time.unique()).sort_values()
        if len(frame) != len(times) * len(SYMBOLS):
            raise AssertionError("incomplete forecast panel")
        frame = frame.sort_values(["time", "symbol"])
        if sorted(SYMBOLS) != list(frame.symbol.unique()):
            raise AssertionError("symbol panel mismatch")
        prices = raw_entry_prices(times)
        n = len(times)
        beta = frame.beta.to_numpy().reshape(n, len(SYMBOLS))
        regime_names = frame.regime.to_numpy().reshape(n, len(SYMBOLS))[:, 0]
        regime = np.array([REGIMES.index(s) for s in regime_names])
        for policy, forecast, field, continuous, threshold_mode, review_hours, switch_hurdle, deadband in CASES:
            pred = frame[field].to_numpy().reshape(n, len(SYMBOLS))
            ent, ext = np.empty(n), np.empty(n)
            months = times.tz_localize(None).to_period("M").astype(str)
            for k in range(n):
                calibration = ("direct" if forecast in ("calm_specialist", "direction_hybrid")
                               and regime_names[k] in ("normal", "high") else forecast)
                ent[k], ext[k] = card_map[(months[k], calibration, regime_names[k], threshold_mode)]
            weights, episode = adaptive_pair_weights(pred, beta, regime, ent, ext,
                continuous=continuous, review_every_hours=review_hours,
                switch_hurdle=switch_hurdle, rebalance_deadband=deadband)
            for fee in (0, 4, 10):
                book = ledger(weights, prices, fee)
                scores.append({"fold": int(fold), "start": str(times[0]), "policy": policy,
                               "fee_bp_side": fee, "hours": n,
                               "total_return": float(book["net_wealth"][-1] - 1),
                               "turnover": float(book["turnover"].sum()),
                               "active_fraction": float((np.abs(weights).sum(axis=1) > .05).mean()),
                               "mean_gross": float(np.abs(weights).sum(axis=1).mean()),
                               "entry_count": int((episode[:, 2] == 1).sum())})
                if fee == 4:
                    gross = np.abs(weights).sum(axis=1)
                    hourly_rows.append(pd.DataFrame({"time": times, "fold": fold, "policy": policy,
                        "regime": regime_names, "net_return": book["net_return"],
                        "gross_return": book["gross_return"], "turnover": book["turnover"],
                        "gross": gross, "age_hours": episode[:, 2]}))
            for state in REGIMES:
                take = regime_names == state
                attribution.append({"fold": int(fold), "policy": policy, "regime": state,
                                    "hours": int(take.sum()),
                                    "active_fraction": float((np.abs(weights[take]).sum(axis=1) > .05).mean()),
                                    "mean_gross": float(np.abs(weights[take]).sum(axis=1).mean())})
        print("completed fixed-forecast policy fold", fold, flush=True)
    scores = pd.DataFrame(scores)
    hourly = pd.concat(hourly_rows, ignore_index=True)
    scores.to_csv(OUT / "policy_scores.csv", index=False)
    pd.DataFrame(attribution).to_csv(OUT / "state_participation.csv", index=False)
    hourly.to_parquet(OUT / "hourly_ledger.parquet", index=False)
    summary = hourly.groupby(["policy", "regime"]).agg(
        hours=("gross", "size"), active=("gross", lambda s: float((s > .05).mean())),
        net_bp_hour=("net_return", lambda s: float(s.mean() * 1e4)),
        turnover=("turnover", "sum")).reset_index()
    summary.to_csv(OUT / "state_outcomes.csv", index=False)
    by_fold = hourly.groupby(["fold", "policy", "regime"]).agg(
        hours=("gross", "size"),
        active=("gross", lambda s: float((s > .05).mean())),
        gross_bp_hour=("gross_return", lambda s: float(s.mean() * 1e4)),
        net_bp_hour=("net_return", lambda s: float(s.mean() * 1e4))).reset_index()
    by_fold.to_csv(OUT / "state_fold_outcomes.csv", index=False)
    daily = hourly.assign(day=hourly.time.dt.floor("D")).groupby(
        ["day", "policy"]).net_return.sum().unstack("policy")
    uncertainty = []
    rng = np.random.default_rng(20260930)
    for name, values in [(c, daily[c].to_numpy()) for c in daily.columns] + [
            ("direction_hybrid_minus_direct", (daily.direction_hybrid_sticky - daily.direct_sticky_continuous).to_numpy()),
            ("calm_specialist_minus_direct", (daily.calm_specialist_sticky - daily.direct_sticky_continuous).to_numpy())]:
        n = len(values)
        draw_start = rng.integers(0, n, (1000, int(np.ceil(n / 14))))
        sample = values[((draw_start[:, :, None] + np.arange(14)) % n).reshape(1000, -1)[:, :n]].mean(axis=1)
        lo, hi = np.quantile(sample, [.025, .975])
        uncertainty.append({"policy_or_difference": name, "days": n,
                            "daily_mean_bp": float(values.mean() * 1e4),
                            "block14_low_bp": float(lo * 1e4),
                            "block14_high_bp": float(hi * 1e4)})
    pd.DataFrame(uncertainty).to_csv(OUT / "policy_uncertainty.csv", index=False)
    shocks = []
    for name, group in hourly.groupby("policy"):
        gross_day = group.assign(day=group.time.dt.floor("D")).groupby("day").net_return.apply(
            lambda s: float(np.prod(1 + s.to_numpy()) - 1))
        top = gross_day.sort_values(ascending=False)
        for n_drop in (0, 1, 3, 5):
            remain = gross_day.drop(top.index[:n_drop])
            shocks.append({"policy": name, "excluded": f"best_{n_drop}_days",
                           "excluded_dates": ",".join(str(t.date()) for t in top.index[:n_drop]),
                           "remaining_compound": float(np.prod(1 + remain.to_numpy()) - 1)})
        shock_days = pd.date_range("2024-12-02", "2024-12-04", tz="UTC")
        remain = gross_day.drop(shock_days)
        shocks.append({"policy": name, "excluded": "2024-12-02_to_04",
                       "excluded_dates": ",".join(str(t.date()) for t in shock_days),
                       "remaining_compound": float(np.prod(1 + remain.to_numpy()) - 1)})
    pd.DataFrame(shocks).to_csv(OUT / "policy_shock_sensitivity.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for policy, group in hourly.groupby("policy"):
        if "hourly" in policy:
            continue
        group = group.sort_values("time")
        axes[0].plot(group.time, np.cumprod(1 + group.net_return), label=policy)
    axes[0].set(title="Fixed-forecast action ablations, 4 bp per side", ylabel="Wealth from 1")
    axes[0].legend(fontsize=7)
    summary[~summary.policy.str.contains("hourly")].pivot(
        index="regime", columns="policy", values="net_bp_hour").reindex(REGIMES).plot.bar(ax=axes[1])
    axes[1].axhline(0, color="black", linewidth=.7)
    axes[1].set(title="Known market state, net bp/hour", xlabel="")
    fig.tight_layout(); fig.savefig(OUT / "policy_ablation.png", dpi=170); plt.close(fig)
    (OUT / "manifest.json").write_text(json.dumps({
        "forecast_source": str(SOURCE / "forward_forecasts.parquet"),
        "threshold_source": str(SOURCE / "threshold_history.csv"),
        "cases": [{"name": c[0], "continuous": c[3], "threshold": c[4],
                   "review_hours": c[5], "switch_hurdle": c[6], "deadband": c[7]} for c in CASES],
        "comparison_status": "exploratory reused development history"},
        indent=2), encoding="utf-8")
    print("saved action iteration", OUT)


if __name__ == "__main__":
    main()
