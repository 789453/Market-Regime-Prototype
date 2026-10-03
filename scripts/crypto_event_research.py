"""Rolling, event-timed crypto research prototype through 2025.

One pooled low-capacity model uses interpretable group representatives from
both 5m and 15m features. Signals update at every completed 15m bar. Execution
is delayed to the next 5m open after that close. 2026 is excluded entirely.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs/crypto_research.yaml").read_text(encoding="utf-8"))
SOURCE = Path(CFG["data"]["root"])
FEATURE_DIR = ROOT / "data/crypto/features"
SIGNAL_DIR = ROOT / "data/crypto/signals"
REPORT_DIR = ROOT / "reports/crypto"
CUTOFF_MS = 1767225600000  # 2026-01-01 UTC; terminal period remains untouched
HORIZON_5M = 48  # 4 hours
SIDE_COST_BPS = (0, 4, 10, 20)  # unverified scenarios
BASE = [
    "momentum_z_med_15m", "momentum_z_med_5m", "efficiency_med_15m",
    "efficiency_med_5m", "rv_fast_slow_15m", "semivar_balance_15m",
    "close_location_5m", "flow_imb_med_15m", "hour_sin_15m", "hour_cos_15m",
    "rv_slow_15m",
]
MODEL_COLUMNS = [
    "trend_15", "trend_5", "efficiency_15", "efficiency_5",
    "vol_intensity", "vol_asymmetry", "path", "flow", "hour_sin", "hour_cos",
    "trend_efficiency", "trend_volatility", "path_flow",
]


def _future_side_variance(log_open: np.ndarray, entry_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    returns = np.diff(log_open, prepend=np.nan)
    returns[0] = 0.0
    up = np.maximum(returns, 0.0) ** 2
    down = np.minimum(returns, 0.0) ** 2
    up_sum = np.r_[0.0, np.cumsum(up)]
    down_sum = np.r_[0.0, np.cumsum(down)]
    start = entry_index + 1
    stop = entry_index + HORIZON_5M + 1
    valid = stop <= len(log_open)
    up_out = np.full(len(entry_index), np.nan)
    down_out = np.full(len(entry_index), np.nan)
    up_out[valid] = up_sum[stop[valid]] - up_sum[start[valid]]
    down_out[valid] = down_sum[stop[valid]] - down_sum[start[valid]]
    return up_out, down_out


def load_symbol(symbol: str) -> pd.DataFrame:
    features = pq.read_table(
        FEATURE_DIR / f"{symbol}.parquet",
        columns=["available_at", *BASE],
        filters=[("available_at", "<", pd.Timestamp("2026-01-01", tz="UTC"))],
    ).to_pandas()
    raw = pq.read_table(
        SOURCE / symbol / "5m.parquet", columns=["open_time", "open"],
        filters=[("open_time", "<", CUTOFF_MS)],
    ).to_pandas()
    open_times = raw.open_time.to_numpy(dtype=np.int64)
    log_open = np.log(raw.open.to_numpy(dtype=float))
    execution_ms = features.available_at.astype("int64").to_numpy() // 1_000_000 + 300_000
    entry = np.searchsorted(open_times, execution_ms)
    aligned = (entry < len(raw)) & (open_times[np.minimum(entry, len(raw) - 1)] == execution_ms)
    if not aligned.all():
        raise ValueError(f"execution time missing in {symbol}")
    valid_exit = entry + HORIZON_5M < len(raw)
    future_ret = np.full(len(entry), np.nan)
    future_ret[valid_exit] = log_open[entry[valid_exit] + HORIZON_5M] - log_open[entry[valid_exit]]
    up, down = _future_side_variance(log_open, entry)
    features["symbol"] = symbol
    features["execution_at"] = pd.to_datetime(execution_ms, unit="ms", utc=True)
    features["label_available_at"] = features.execution_at + pd.Timedelta(hours=4)
    features["execution_price"] = raw.open.to_numpy(dtype=float)[entry]
    features["future_logret"] = future_ret
    features["future_up_var"] = up
    features["future_down_var"] = down
    sigma_4h = np.sqrt(features.rv_slow_15m / 6).clip(lower=1e-5)
    features["sigma_4h"] = sigma_4h
    features["target_z"] = (future_ret / sigma_4h).clip(-10, 10)
    features["trend_15"] = features.momentum_z_med_15m.clip(-8, 8)
    features["trend_5"] = features.momentum_z_med_5m.clip(-8, 8)
    features["efficiency_15"] = features.efficiency_med_15m.clip(0, 1)
    features["efficiency_5"] = features.efficiency_med_5m.clip(0, 1)
    features["vol_intensity"] = np.log(features.rv_fast_slow_15m.clip(lower=1e-5, upper=100))
    features["vol_asymmetry"] = features.semivar_balance_15m.clip(-1, 1)
    features["path"] = features.close_location_5m.clip(-1, 1)
    features["flow"] = features.flow_imb_med_15m.clip(-1, 1)
    features["hour_sin"] = features.hour_sin_15m
    features["hour_cos"] = features.hour_cos_15m
    features["trend_efficiency"] = features.trend_15 * features.efficiency_15
    features["trend_volatility"] = features.trend_15 * features.vol_intensity
    features["path_flow"] = features.path * features.flow
    features = features.replace([np.inf, -np.inf], np.nan)
    return features[["symbol", "available_at", "execution_at", "label_available_at", "execution_price", "rv_slow_15m", "sigma_4h", "future_logret", "future_up_var", "future_down_var", "target_z", *MODEL_COLUMNS]].dropna().reset_index(drop=True)


def rolling_predictions(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = pd.Timestamp("2026-01-01", tz="UTC")
    starts = list(pd.date_range(pd.Timestamp("2023-07-01", tz="UTC"), cutoff, freq="30D"))
    if starts[-1] < cutoff:
        starts.append(cutoff)
    pieces = []
    fold_rows = []
    for fold, start in enumerate(starts[:-1]):
        end = starts[fold + 1]
        train = panel.loc[
            (panel.available_at >= start - pd.Timedelta(days=180))
            & (panel.label_available_at < start)
            & (panel.available_at.dt.minute == 0)
        ]
        test = panel.loc[(panel.available_at >= start) & (panel.available_at < end)].copy()
        if len(train) < 20_000 or test.empty:
            continue
        scaler = StandardScaler().fit(train[MODEL_COLUMNS])
        x_train = np.clip(scaler.transform(train[MODEL_COLUMNS]), -8, 8)
        x_test = np.clip(scaler.transform(test[MODEL_COLUMNS]), -8, 8)
        return_model = Ridge(alpha=1000).fit(x_train, train.target_z)
        test["pred_z"] = return_model.predict(x_test)
        risk_stats = {}
        for side in ("up", "down"):
            label = np.log(train[f"future_{side}_var"].clip(lower=1e-10))
            risk_model = Ridge(alpha=1000).fit(x_train, label)
            actual = np.log(test[f"future_{side}_var"].clip(lower=1e-10))
            predicted = risk_model.predict(x_test)
            test[f"pred_{side}_var"] = np.exp(predicted)
            base_mse = float(np.mean((actual - label.mean()) ** 2))
            model_mse = float(np.mean((actual - predicted) ** 2))
            naive = np.log((test.rv_slow_15m / 12).clip(lower=1e-10))
            naive_mse = float(np.mean((actual - naive) ** 2))
            risk_stats[f"{side}_risk_oos_r2"] = 1 - model_mse / base_mse
            risk_stats[f"{side}_risk_skill_vs_recent_rv"] = 1 - model_mse / naive_mse
        test["pred_logret"] = test.pred_z * test.sigma_4h
        # A round trip at the 10 bp/side scenario is a 20 bp hurdle.
        test["target_position"] = np.where(test.pred_logret > 0.002, 1.0, np.where(test.pred_logret < -0.002, -1.0, 0.0))
        corr = test[["pred_z", "target_z"]].corr().iloc[0, 1]
        fold_rows.append({
            "fold": fold, "start": str(start), "end": str(end),
            "train_rows": len(train), "test_rows": len(test),
            "return_ic": float(corr) if np.isfinite(corr) else None,
            "return_oos_r2": 1 - float(np.mean((test.target_z - test.pred_z) ** 2)) / float(np.mean((test.target_z - train.target_z.mean()) ** 2)),
            "active_fraction": float((test.target_position != 0).mean()),
            **risk_stats,
        })
        pieces.append(test)
    if not pieces:
        raise RuntimeError("no rolling folds were produced")
    return pd.concat(pieces, ignore_index=True), pd.DataFrame(fold_rows)


def add_hysteresis(predictions: pd.DataFrame) -> pd.DataFrame:
    """Enter above cost hurdle; hold while directional forecast stays signed."""
    predictions = predictions.copy()
    predictions["hysteresis_position"] = 0.0
    for _, frame in predictions.groupby("symbol", sort=False):
        ordered = frame.sort_values("execution_at")
        positions = np.zeros(len(ordered))
        current = 0.0
        for i, prediction in enumerate(ordered.pred_logret.to_numpy()):
            if prediction > 0.002:
                current = 1.0
            elif prediction < -0.002:
                current = -1.0
            elif current * prediction < 0:
                current = 0.0
            positions[i] = current
        predictions.loc[ordered.index, "hysteresis_position"] = positions
    return predictions


def evaluate_events(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = []
    daily_records = []
    # Positions change only at signal-triggered executions, not at a fixed UTC hour.
    for bp in SIDE_COST_BPS:
        for position_field in ("target_position", "hysteresis_position"):
            streams = []
            for _, frame in predictions.groupby("symbol", sort=False):
                frame = frame.sort_values("execution_at")
                target = np.ones(len(frame)) if position_field == "buyhold" else frame[position_field].to_numpy()
                old_position = np.r_[0.0, target[:-1]]
                price = frame.execution_price.to_numpy()
                price_change = price / np.r_[price[0], price[:-1]] - 1
                turnover = np.abs(target - old_position)
                streams.append(pd.DataFrame({
                    "execution_at": frame.execution_at,
                    "return": old_position * price_change - turnover * bp / 10000,
                    "turnover": turnover, "active": target != 0,
                }))
            bars = pd.concat(streams).groupby("execution_at", sort=True)[["return", "turnover", "active"]].mean()
            daily = (1 + bars["return"]).groupby(bars.index.floor("D")).prod() - 1
            equity = (1 + daily).cumprod()
            daily_records.extend({"date": str(day.date()), "strategy": position_field, "side_cost_bp": bp, "daily_return": float(value)} for day, value in daily.items())
            result.append({
                "strategy": position_field, "side_cost_bp": bp,
                "days": len(daily), "mean_daily_bp": float(daily.mean() * 10000),
                "annualized_sharpe": float(daily.mean() / daily.std(ddof=1) * np.sqrt(365)),
                "cumulative_return": float(equity.iloc[-1] - 1),
                "max_drawdown": float((equity / equity.cummax() - 1).min()),
                "mean_active_fraction": float(bars.active.mean()),
                "total_turnover_per_symbol_mean": float(bars.turnover.sum()),
            })
        # True buy-and-hold: equal initial cash allocation, no intermediate
        # rebalance, one entry and one terminal exit per symbol.
        prices = predictions.pivot(index="execution_at", columns="symbol", values="execution_price").sort_index().ffill().dropna()
        relative = prices / prices.iloc[0]
        fee = bp / 10000
        wealth = relative.mean(axis=1) - fee
        wealth.iloc[-1] -= fee * relative.iloc[-1].mean()
        daily_wealth = wealth.groupby(wealth.index.floor("D")).last()
        daily = daily_wealth.pct_change()
        daily.iloc[0] = daily_wealth.iloc[0] - 1
        equity = (1 + daily).cumprod()
        daily_records.extend({"date": str(day.date()), "strategy": "buyhold", "side_cost_bp": bp, "daily_return": float(value)} for day, value in daily.items())
        result.append({
            "strategy": "buyhold", "side_cost_bp": bp,
            "days": len(daily), "mean_daily_bp": float(daily.mean() * 10000),
            "annualized_sharpe": float(daily.mean() / daily.std(ddof=1) * np.sqrt(365)),
            "cumulative_return": float(equity.iloc[-1] - 1),
            "max_drawdown": float((equity / equity.cummax() - 1).min()),
            "mean_active_fraction": 1.0,
            "total_turnover_per_symbol_mean": 2.0,
        })
    return pd.DataFrame(result), pd.DataFrame(daily_records)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    panel = pd.concat([load_symbol(symbol) for symbol in CFG["data"]["symbols"]], ignore_index=True)
    predictions, folds = rolling_predictions(panel)
    predictions = add_hysteresis(predictions)
    signals = predictions[["symbol", "available_at", "execution_at", "execution_price", "pred_z", "pred_logret", "pred_up_var", "pred_down_var", "target_position", "hysteresis_position"]]
    signals.to_parquet(SIGNAL_DIR / "walkforward_event.parquet", index=False, compression="zstd")
    folds.to_csv(REPORT_DIR / "event_walkforward_folds.csv", index=False)
    scores, daily = evaluate_events(predictions)
    scores.to_csv(REPORT_DIR / "event_walkforward_scores.csv", index=False)
    daily.to_csv(REPORT_DIR / "event_walkforward_daily.csv", index=False)
    metadata = {
        "date_limit": "2025-12-31; 2026 rows excluded before modelling",
        "features": MODEL_COLUMNS, "base_features": BASE,
        "train_days": 180, "test_days": 30, "train_sample_frequency": "1h", "signal_frequency": "15m",
        "label_horizon_hours": 4, "execution_delay_minutes": 5,
        "ridge_alpha": 1000, "entry_hurdle_abs_logret": 0.002,
        "hysteresis_exit": "exit when directional prediction changes sign; strong opposite signal reverses",
        "side_cost_bp_scenarios": SIDE_COST_BPS,
        "funding": "missing; not included", "benchmark": "buy and hold only",
        "folds": len(folds), "signals": len(signals),
    }
    (REPORT_DIR / "event_walkforward_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))
    print(scores.to_string(index=False))
    print(folds[["return_ic", "return_oos_r2", "up_risk_oos_r2", "down_risk_oos_r2", "up_risk_skill_vs_recent_rv", "down_risk_skill_vs_recent_rv", "active_fraction"]].mean().to_string())


if __name__ == "__main__":
    main()
