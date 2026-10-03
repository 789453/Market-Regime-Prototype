"""Exploratory daily decision pilot for the crypto state research program.

Uses hourly closed bars, decides at 23:00 UTC close, enters at next 00:00
open, and exits/updates at the following 00:00 open. 2026 is untouched.
No fee or funding data is claimed; cost values are sensitivity scenarios.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs/crypto_research.yaml").read_text(encoding="utf-8"))
OUT = ROOT / "reports/crypto"
FEATURES = ["trend_z", "efficiency", "vol_ratio", "flow_imbalance", "activity_ratio"]
COST_BPS_PER_SIDE = (0, 4, 10, 20)  # scenarios, not measured fees


def symbol_frame(symbol: str) -> pd.DataFrame:
    path = Path(CFG["data"]["root"]) / symbol / "1h.parquet"
    # The pilot never loads 2026 rows into its research frame.
    df = pq.read_table(
        path,
        columns=["open_time", "open", "close", "quote_volume", "taker_buy_quote_volume"],
        filters=[("open_time", "<", 1767225600000)],  # 2026-01-01 00:00 UTC
    ).to_pandas()
    dt = pd.to_datetime(df.open_time, unit="ms", utc=True)
    ret = np.log(df.close).diff()
    vol = ret.rolling(168, min_periods=168).std().shift(1)
    momentum = np.log(df.close / df.close.shift(72))
    df["trend_z"] = (momentum / (vol * np.sqrt(72))).clip(-8, 8)
    df["efficiency"] = (momentum.abs() / ret.abs().rolling(72, min_periods=72).sum()).clip(0, 1)
    df["vol_ratio"] = (ret.rolling(24, min_periods=24).std() / vol).clip(0, 8)
    q24 = df.quote_volume.rolling(24, min_periods=24).sum()
    buy24 = df.taker_buy_quote_volume.rolling(24, min_periods=24).sum()
    df["flow_imbalance"] = (2 * buy24 / q24 - 1).clip(-1, 1)
    df["activity_ratio"] = (q24 / (df.quote_volume.rolling(168, min_periods=168).mean().shift(1) * 24)).clip(0, 20)
    df["entry"] = df.open.shift(-1)
    df["exit"] = df.open.shift(-25)
    df["future_ret"] = np.log(df["exit"] / df["entry"])
    df["symbol"] = symbol
    df["decision_time"] = dt + pd.Timedelta(hours=1)
    df["exit_time"] = dt + pd.Timedelta(hours=25)
    df = df.loc[dt.dt.hour == 23, ["symbol", "decision_time", "exit_time", *FEATURES, "future_ret"]]
    return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def portfolio(daily: pd.DataFrame, position: str, start: str, end: str) -> dict:
    frame = daily.loc[(daily.decision_time >= start) & (daily.exit_time < end)].copy()
    frame.sort_values(["symbol", "decision_time"], inplace=True)
    frame["position"] = frame[position]
    frame["turnover"] = (frame.position - frame.groupby("symbol").position.shift(fill_value=0)).abs()
    frame["gross"] = frame.position * np.expm1(frame.future_ret)
    grouped = frame.groupby("decision_time", sort=True)[["gross", "turnover"]].mean()
    last_year = pd.Timestamp(end).year - 1
    period = start[:4] if int(start[:4]) == last_year else f"{start[:4]}-{last_year}"
    result = {"strategy": position, "period": period, "days": len(grouped), "trades": int((frame.turnover > 0).sum()), "mean_abs_position": float(frame.position.abs().mean())}
    for bp in COST_BPS_PER_SIDE:
        net = grouped.gross - bp / 10000 * grouped.turnover
        result[f"mean_daily_bp_cost_{bp}"] = float(net.mean() * 10000)
        result[f"sharpe_cost_{bp}"] = float(net.mean() / net.std(ddof=1) * np.sqrt(365)) if net.std(ddof=1) > 0 else None
        result[f"cum_log_equity_cost_{bp}"] = float(np.log1p(net).sum()) if (net > -1).all() else None
        if bp == 10:
            values = net.to_numpy()
            rng = np.random.default_rng(20260928)
            starts = rng.integers(0, len(values), size=(2000, int(np.ceil(len(values) / 7))))
            indices = ((starts[:, :, None] + np.arange(7)) % len(values)).reshape(2000, -1)[:, :len(values)]
            means = values[indices].mean(axis=1) * 10000
            result["block7_ci_low_bp_cost_10"] = float(np.quantile(means, 0.025))
            result["block7_ci_high_bp_cost_10"] = float(np.quantile(means, 0.975))
    return result


def state_block_intervals(data: pd.DataFrame) -> pd.DataFrame:
    """Seven-day circular block intervals; assets on a day move together."""
    valid = data.loc[(data.decision_time >= "2025-01-01") & (data.exit_time < "2026-01-01")].copy()
    valid["day"] = valid.decision_time.dt.floor("D")
    days = pd.Index(sorted(valid.day.unique()))
    grouped = valid.groupby(["day", "state"]).signed_future_ret.agg(["sum", "count"])
    rng = np.random.default_rng(20260928)
    n = len(days)
    draws = 2000
    starts = rng.integers(0, n, size=(draws, int(np.ceil(n / 7))))
    offsets = np.arange(7)
    indices = ((starts[:, :, None] + offsets) % n).reshape(draws, -1)[:, :n]
    rows = []
    for state in sorted(valid.state.unique()):
        state_daily = grouped.xs(state, level="state").reindex(days, fill_value=0)
        sums = state_daily["sum"].to_numpy()
        counts = state_daily["count"].to_numpy()
        sampled_count = counts[indices].sum(axis=1)
        sampled_mean = np.divide(sums[indices].sum(axis=1), sampled_count, out=np.full(draws, np.nan), where=sampled_count > 0)
        rows.append({
            "state": int(state), "observations": int(counts.sum()),
            "distinct_days": int((counts > 0).sum()),
            "mean_signed_future_bp": float(sums.sum() / counts.sum() * 10000),
            "block7_ci_low_bp": float(np.nanquantile(sampled_mean, 0.025) * 10000),
            "block7_ci_high_bp": float(np.nanquantile(sampled_mean, 0.975) * 10000),
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.concat([symbol_frame(s) for s in CFG["data"]["symbols"]], ignore_index=True)
    train = data.loc[data.exit_time < "2025-01-01"].copy()
    valid = data.loc[(data.decision_time >= "2025-01-01") & (data.exit_time < "2026-01-01")].copy()
    development = pd.concat([train, valid], ignore_index=True)
    scaler = StandardScaler().fit(train[FEATURES])
    model = KMeans(n_clusters=6, random_state=20260928, n_init=20).fit(scaler.transform(train[FEATURES]))
    development["state"] = model.predict(scaler.transform(development[FEATURES]))
    train_mask = development.exit_time < "2025-01-01"
    development["direction"] = np.sign(development.trend_z)
    development["signed_future_ret"] = development.direction * development.future_ret
    # One globally shared state weight, estimated only on 2023-24. This is
    # deliberately coarse; 2025 measures whether the conditional edge transfers.
    state_train = development.loc[train_mask].groupby("state").signed_future_ret.agg(["mean", "count"])
    state_train["weight"] = np.where(state_train["mean"] > 0, 1.0, 0.0)
    development["baseline_momentum"] = development.direction
    development["always_long"] = 1.0
    development["state_filtered"] = development.direction * development.state.map(state_train.weight)
    # No outcome fit: economic interpretation is persistent directional flow.
    development["flow_confirmed"] = development.direction * (
        (development.efficiency >= float(train.efficiency.median()))
        & (development.flow_imbalance * development.direction > 0)
    ).astype(float)
    scores = []
    for name in ("always_long", "baseline_momentum", "state_filtered", "flow_confirmed"):
        scores.append(portfolio(development, name, "2023-01-01", "2025-01-01"))
        scores.append(portfolio(development, name, "2025-01-01", "2026-01-01"))
    pd.DataFrame(scores).to_csv(OUT / "pilot_strategy_scores.csv", index=False)
    state = development.groupby([development.decision_time.dt.year.rename("year"), "state"]).agg(
        observations=("signed_future_ret", "size"),
        distinct_days=("decision_time", "nunique"),
        mean_signed_future_bp=("signed_future_ret", lambda x: float(x.mean() * 10000)),
        mean_trend_z=("trend_z", "mean"),
        mean_efficiency=("efficiency", "mean"),
        mean_vol_ratio=("vol_ratio", "mean"),
        mean_flow_imbalance=("flow_imbalance", "mean"),
        mean_activity_ratio=("activity_ratio", "mean"),
    ).reset_index()
    state.to_csv(OUT / "pilot_state_cards.csv", index=False)
    detail = development.groupby([
        development.decision_time.dt.year.rename("year"), "symbol", "state", "direction"
    ]).agg(
        observations=("signed_future_ret", "size"),
        mean_signed_future_bp=("signed_future_ret", lambda x: float(x.mean() * 10000)),
    ).reset_index()
    detail.to_csv(OUT / "pilot_state_symbol_direction.csv", index=False)
    state_block_intervals(development).to_csv(OUT / "pilot_validation_block_intervals.csv", index=False)
    metadata = {
        "train_rows": len(train), "validation_rows": len(valid),
        "train_period": "2023-2024", "validation_period": "2025", "untouched_period": "2026",
        "state_weights": {str(k): float(v) for k, v in state_train.weight.items()},
        "feature_names": FEATURES, "cost_bps_per_side_scenarios": COST_BPS_PER_SIDE,
        "decision": "23:00 UTC bar close", "entry": "following 00:00 UTC open",
        "exit": "next day's 00:00 UTC open", "funding": "not included; unavailable",
    }
    (OUT / "pilot_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))
    print(pd.DataFrame(scores).to_string(index=False))


if __name__ == "__main__":
    main()
