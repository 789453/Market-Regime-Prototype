"""First complete, exploratory state -> forecast -> action -> multi-asset ledger."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.crypto_predictive_states_stage import CFG, SYMBOLS, load_panel
from src.crypto.predictive_states import make_representation
from src.crypto.predictive_states_v2 import center_membership, compact_features
from src.crypto.signal_chain import (beta_neutral_pair, buy_hold, ledger,
                                     market_weights, pair_weights)

OUT = ROOT / "reports/crypto/signal_chain_v1"
V2 = ROOT / "reports/crypto/predictive_states_v2"
SEED = 20260929


def read_past_beta(times: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """30-day rolling beta from completed 15m returns; also past-day momentum."""
    returns, closes = [], []
    for symbol in SYMBOLS:
        table = pq.read_table(Path(CFG["source"]) / symbol / "15m.parquet",
                              columns=["open_time", "log_return", "close"]).to_pandas()
        if len(table) < len(times):
            raise AssertionError(f"raw 15m shorter than feature panel for {symbol}")
        table = table.iloc[:len(times)]
        available = pd.to_datetime(table.open_time, unit="ms", utc=True) + pd.Timedelta(minutes=15)
        if not np.array_equal(available.to_numpy(), times.to_numpy()):
            raise AssertionError(f"15m raw/feature clock mismatch for {symbol}")
        returns.append(np.nan_to_num(table.log_return.to_numpy(dtype=np.float64)))
        closes.append(np.log(table.close.to_numpy(dtype=np.float64)))
    r = np.stack(returns, axis=1)
    m = r.mean(axis=1)
    ms = pd.Series(m)
    win, minimum = 96 * 30, 96 * 7
    mm = ms.rolling(win, min_periods=minimum).mean().to_numpy()
    var = pd.Series(m * m).rolling(win, min_periods=minimum).mean().to_numpy() - mm * mm
    beta = np.empty_like(r)
    for j in range(len(SYMBOLS)):
        a = pd.Series(r[:, j])
        ar = a.rolling(win, min_periods=minimum).mean().to_numpy()
        cross = pd.Series(r[:, j] * m).rolling(win, min_periods=minimum).mean().to_numpy()
        beta[:, j] = np.clip((cross - ar * mm) / np.maximum(var, 1e-12), .2, 2.5)
    close = np.stack(closes, axis=1)
    momentum = np.full_like(close, np.nan)
    momentum[96:] = close[96:] - close[:-96]
    market_30d = np.full(len(close), np.nan)
    market_30d[win:] = (close[win:] - close[:-win]).mean(axis=1)
    market_risk = np.sqrt(np.maximum(var, 0) * win)
    return beta, momentum, market_30d, market_risk


def fit_regression(x: np.ndarray, y: np.ndarray, *, market: bool, seed: int) -> lgb.LGBMRegressor:
    low, high = np.quantile(y, [.005, .995])
    model = lgb.LGBMRegressor(n_estimators=85 if market else 100,
                             learning_rate=.035, num_leaves=5 if market else 7,
                             max_depth=3 if market else 4,
                             min_child_samples=35 if market else 160,
                             reg_lambda=25, colsample_bytree=.85, n_jobs=6,
                             verbosity=-1, random_state=seed)
    model.fit(x, np.clip(y, low, high))
    return model


def interval_price(times: pd.Series) -> np.ndarray:
    """Exact next-5m open at each daily decision and the terminal mark."""
    asks = pd.DatetimeIndex(times + pd.Timedelta(minutes=5))
    asks = asks.append(pd.DatetimeIndex([asks[-1] + pd.Timedelta(days=1)]))
    requested = asks.asi8 // 1_000_000
    prices = np.empty((len(asks), len(SYMBOLS)), dtype=np.float64)
    for j, symbol in enumerate(SYMBOLS):
        raw = pq.read_table(Path(CFG["source"]) / symbol / "5m.parquet",
                            columns=["open_time", "open"],
                            filters=[("open_time", ">=", int(requested[0])),
                                     ("open_time", "<=", int(requested[-1]))]).to_pandas()
        stamp = raw.open_time.to_numpy(dtype=np.int64)
        pos = np.searchsorted(stamp, requested)
        if np.any(pos >= len(stamp)) or not np.array_equal(stamp[pos], requested):
            raise AssertionError(f"missing exact execution open for {symbol}")
        prices[:, j] = np.log(raw.open.to_numpy(dtype=np.float64)[pos])
    return prices


def block_interval(daily: np.ndarray, seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(daily)
    start = rng.integers(0, n, (1200, int(np.ceil(n / 7))))
    draw = daily[((start[:, :, None] + np.arange(7)) % n).reshape(1200, -1)[:, :n]].mean(axis=1)
    return tuple(np.quantile(draw, [.025, .975]))


def prediction_audit(phase: str, horizon: int, time: pd.Series,
                     market_y: np.ndarray, market_p: np.ndarray,
                     residual_y: np.ndarray,
                     forecasts: dict[str, np.ndarray]) -> tuple[list[dict], pd.DataFrame]:
    rows = []
    market_ic = pd.Series(market_y).corr(pd.Series(market_p), method="spearman")
    rows.append({"phase": phase, "horizon_h": horizon, "target": "market", "model": "continuous",
                 "spearman": market_ic, "mse": np.mean((market_p - market_y) ** 2),
                 "signed_top_bottom": np.nan})
    buckets = []
    for name, pred in forecasts.items():
        daily_ic = [pd.Series(pred[k]).corr(pd.Series(residual_y[k]), method="spearman")
                    for k in range(len(pred))]
        spread = []
        for k in range(len(pred)):
            order = np.argsort(pred[k])
            spread.append(residual_y[k, order[-2:]].mean() - residual_y[k, order[:2]].mean())
        rows.append({"phase": phase, "horizon_h": horizon, "target": "coin_residual",
                     "model": name, "spearman": np.nanmean(daily_ic),
                     "mse": np.mean((pred - residual_y) ** 2),
                     "signed_top_bottom": np.mean(spread),
                     "spread_block_low": block_interval(np.asarray(spread))[0],
                     "spread_block_high": block_interval(np.asarray(spread))[1]})
        frame = pd.DataFrame({"score": pred.ravel(), "actual": residual_y.ravel(),
                              "time": np.repeat(time.to_numpy(), pred.shape[1])})
        frame["quantile"] = pd.qcut(frame.score.rank(method="first"), 5, labels=False)
        group = frame.groupby("quantile").actual.agg(["mean", "count",
                   lambda x: x.quantile(.05), lambda x: x.quantile(.95)]).reset_index()
        group.columns = ["quantile", "mean", "count", "q05", "q95"]
        group["phase"], group["horizon_h"], group["model"] = phase, horizon, name
        buckets.append(group)
    return rows, pd.concat(buckets, ignore_index=True)


def run_stage(data: dict, train_end: str, eval_start: str, eval_end: str,
              phase: str) -> tuple[list[dict], list[pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    time = data["time"]
    train_end_t = pd.Timestamp(train_end, tz="UTC")
    begin_t, end_t = pd.Timestamp(eval_start, tz="UTC"), pd.Timestamp(eval_end, tz="UTC")
    train = np.flatnonzero((time + pd.Timedelta(days=1, minutes=5) < train_end_t).to_numpy()
                           & (time >= pd.Timestamp("2023-02-01", tz="UTC")).to_numpy())
    evaluate = np.flatnonzero((time >= begin_t).to_numpy() & (time < end_t).to_numpy())
    if len(train) < 300 or len(evaluate) < 30:
        raise AssertionError("insufficient chronology")
    n = len(time)
    train_coin = np.concatenate([train + j * n for j in range(len(SYMBOLS))])
    eval_coin = np.concatenate([evaluate + j * n for j in range(len(SYMBOLS))])
    all_pred, audit, quantile_cards = {}, [], []
    for horizon in (4, 24):
        m_y = data[f"market_y{horizon}"]
        a_y = data[f"alpha_y{horizon}"]
        model_m = fit_regression(data["market_x"][train], m_y[train], market=True,
                                 seed=SEED + horizon)
        pred_m = model_m.predict(data["market_x"][evaluate])
        pred_a = {}
        for name, x in (("base", data["coin_x"]), ("geometry", data["coin_geom_x"]),
                        ("full_x2", data["coin_full_x"]),
                        ("full_x2_own", data["coin_full_own_x"])):
            model = fit_regression(x[train_coin], a_y.T.ravel()[train_coin], market=False,
                                   seed=SEED + horizon + (0 if name == "base" else 100))
            pred_a[name] = model.predict(x[eval_coin]).reshape(len(SYMBOLS), -1).T
            if phase == "2026_exploratory":
                joblib.dump(model, OUT / f"{name}_alpha_{horizon}h.joblib", compress=3)
        if phase == "2026_exploratory":
            joblib.dump(model_m, OUT / f"market_{horizon}h.joblib", compress=3)
        rows, buckets = prediction_audit(phase, horizon, time.iloc[evaluate].reset_index(drop=True),
                                         m_y[evaluate], pred_m, a_y[evaluate],
                                         {"base": pred_a["base"],
                                          "plus_k48_distance": pred_a["geometry"],
                                          "full_x2_market": pred_a["full_x2"],
                                          "full_x2_own": pred_a["full_x2_own"]})
        audit += rows
        quantile_cards.append(buckets)
        all_pred[horizon] = (pred_m, pred_a)
    eval_time = time.iloc[evaluate].reset_index(drop=True)
    px = interval_price(eval_time)
    # All evaluation dates form one regular daily grid.
    if not (eval_time.diff().iloc[1:] == pd.Timedelta(days=1)).all():
        raise AssertionError("decision clock has gaps")
    market_p24, alpha_p24 = all_pred[24]
    train_market_model = fit_regression(data["market_x"][train], data["market_y24"][train],
                                        market=True, seed=SEED + 24)
    train_market_p = train_market_model.predict(data["market_x"][train])
    market_enter, market_leave = np.quantile(np.abs(train_market_p), [.70, .35])
    pair_quantiles = {}
    for name, x in (("base", data["coin_x"]), ("geometry", data["coin_geom_x"]),
                    ("full_x2", data["coin_full_x"]),
                    ("full_x2_own", data["coin_full_own_x"])):
        model = fit_regression(x[train_coin], data["alpha_y24"].T.ravel()[train_coin],
                               market=False, seed=SEED + 24 + (0 if name == "base" else 100))
        train_p = model.predict(x[train_coin]).reshape(len(SYMBOLS), -1).T
        pair_quantiles[name] = np.ptp(train_p, axis=1)
    pair_enter, pair_leave = np.quantile(pair_quantiles["base"], [.70, .35])
    geom_enter, geom_leave = np.quantile(pair_quantiles["geometry"], [.70, .35])
    full_enter, full_leave = np.quantile(pair_quantiles["full_x2"], [.70, .35])
    own_enter, own_leave = np.quantile(pair_quantiles["full_x2_own"], [.70, .35])
    risk = data["risk_p"][evaluate]
    beta = data["beta"][evaluate]
    momentum = data["momentum"][evaluate]
    train_momentum = data["momentum"][train]
    momentum_market = np.mean(momentum, axis=1)
    momentum_train = np.mean(train_momentum, axis=1)
    mom_market_enter, mom_market_leave = np.quantile(np.abs(momentum_train), [.70, .35])
    mom_pair_enter, mom_pair_leave = np.quantile(
        np.ptp(train_momentum, axis=1), [.70, .35])
    weights = {
        "market_base": market_weights(market_p24, market_enter, market_leave),
        "market_risk_gate": market_weights(market_p24, market_enter, market_leave,
                                           risk=risk.mean(axis=1)),
        "pair_base": pair_weights(alpha_p24["base"], beta, pair_enter, pair_leave),
        "pair_k48_distance": pair_weights(alpha_p24["geometry"], beta, geom_enter, geom_leave),
        "pair_full_x2_direct": pair_weights(alpha_p24["full_x2"], beta, full_enter, full_leave),
        "pair_full_x2_sparse_posthoc": pair_weights(
            alpha_p24["full_x2"], beta, *np.quantile(pair_quantiles["full_x2"], [.80, .40])),
        "pair_full_x2_own": pair_weights(alpha_p24["full_x2_own"], beta, own_enter, own_leave),
        "pair_full_x2_own_sparse_posthoc": pair_weights(
            alpha_p24["full_x2_own"], beta, *np.quantile(pair_quantiles["full_x2_own"], [.80, .40])),
        "market_past_momentum": market_weights(momentum_market, mom_market_enter, mom_market_leave),
        "pair_past_momentum": pair_weights(momentum, beta, mom_pair_enter, mom_pair_leave),
    }
    weights["pair_reverse_full"] = -weights["pair_full_x2_direct"]
    weights["pair_reverse_sparse"] = -weights["pair_full_x2_sparse_posthoc"]
    weights["pair_reverse_own_sparse"] = -weights["pair_full_x2_own_sparse_posthoc"]
    same_entry = np.zeros_like(weights["pair_full_x2_direct"])
    active_full = np.abs(weights["pair_full_x2_direct"]).sum(axis=1) > 0
    for t in np.flatnonzero(active_full):
        same_entry[t] = beta_neutral_pair(int(np.argmax(momentum[t])),
                                          int(np.argmin(momentum[t])), beta[t])
    weights["pair_momentum_same_entry"] = same_entry
    sparse_same = np.zeros_like(weights["pair_full_x2_sparse_posthoc"])
    for t in np.flatnonzero(np.abs(weights["pair_full_x2_sparse_posthoc"]).sum(axis=1) > 0):
        sparse_same[t] = beta_neutral_pair(int(np.argmax(momentum[t])),
                                           int(np.argmin(momentum[t])), beta[t])
    weights["pair_momentum_sparse_same_entry"] = sparse_same
    own_same = np.zeros_like(weights["pair_full_x2_own_sparse_posthoc"])
    for t in np.flatnonzero(np.abs(weights["pair_full_x2_own_sparse_posthoc"]).sum(axis=1) > 0):
        own_same[t] = beta_neutral_pair(int(np.argmax(momentum[t])),
                                        int(np.argmin(momentum[t])), beta[t])
    weights["pair_momentum_own_sparse_same_entry"] = own_same
    benchmark = buy_hold(px)
    bench_returns = np.diff(benchmark) / benchmark[:-1]
    ledger_rows, positions, score_rows = [], [], []
    market_simple = np.expm1(np.diff(px, axis=0)).mean(axis=1)
    for name, w in weights.items():
        gross = ledger(w, px, 0)
        gross_exposure = np.abs(w).sum(axis=1)
        old = np.vstack((np.zeros((1, 12)), w[:-1]))
        if name.startswith("market"):
            key = np.sign(w.sum(axis=1)).astype(int)
        else:
            key = np.where(gross_exposure > 0,
                           1 + 12 * np.argmax(w, axis=1) + np.argmin(w, axis=1), 0)
        previous = np.r_[0, key[:-1]]
        opened = (key != 0) & (key != previous)
        run_end = np.flatnonzero(np.r_[key[1:] != key[:-1], True]) + 1
        run_start = np.r_[0, run_end[:-1]]
        hold = (run_end - run_start)[key[run_start] != 0]
        for fee in (0, 4, 10):
            result = ledger(w, px, fee)
            result_daily = result["net_return"]
            market_part = (w * beta).sum(axis=1) * market_simple
            score_rows.append({"phase": phase, "strategy": name, "fee_bp_side": fee,
                               "days": len(eval_time), "total_return": result["net_wealth"][-1] - 1,
                               "buy_hold_total_return": benchmark[-1] - 1,
                               "mean_bp_day": result_daily.mean() * 1e4,
                               "sharpe_daily_ann": np.sqrt(365) * result_daily.mean() /
                                   max(result_daily.std(ddof=1), 1e-10),
                               "max_drawdown": np.min(result["net_wealth"] /
                                   np.maximum.accumulate(np.r_[1, result["net_wealth"]])[1:] - 1),
                               "active_fraction": np.mean(gross_exposure > 0),
                               "open_events": int(opened.sum()),
                               "order_legs": int(np.sum(np.abs(w - old) > 1e-9)),
                               "avg_hold_hours": float(24 * hold.mean()) if len(hold) else 0,
                               "gross_turnover": gross["turnover"].sum(),
                               "beta_contribution": market_part.sum(),
                               "residual_contribution": (gross["gross_return"] - market_part).sum()})
            ledger_rows.append(pd.DataFrame({"available_at": eval_time, "phase": phase,
                "strategy": name, "fee_bp_side": fee, "gross_return": gross["gross_return"],
                "net_return": result_daily, "gross_wealth": gross["gross_wealth"],
                "net_wealth": result["net_wealth"], "turnover": gross["turnover"],
                "gross_exposure": gross_exposure, "market_beta_exposure": (w * beta).sum(axis=1),
                "market_component": market_part, "residual_component": gross["gross_return"] - market_part,
                "buy_hold_wealth": benchmark[1:], "buy_hold_return": bench_returns,
                "market_pred24": market_p24, "market_actual24": data["market_y24"][evaluate],
                "mean_high_vol_p": risk.mean(axis=1)}))
        pos = pd.DataFrame({"available_at": np.repeat(eval_time.to_numpy(), 12),
                            "symbol": np.tile(SYMBOLS, len(eval_time)),
                            "weight": w.ravel(), "beta_past": beta.ravel(),
                            "asset_contribution": gross["asset_contribution"].ravel(),
                            "coin_buy_hold_wealth": np.exp(px[1:] - px[0]).ravel(),
                            "strategy": name, "phase": phase})
        positions.append(pos)
    frame = pd.DataFrame({"available_at": np.repeat(eval_time.to_numpy(), 12),
                          "symbol": np.tile(SYMBOLS, len(eval_time)),
                          "phase": phase, "beta_past": beta.ravel(),
                          "return4": data["return4"][evaluate].ravel(),
                          "return24": data["return24"][evaluate].ravel(),
                          "alpha4": data["alpha_y4"][evaluate].ravel(),
                          "alpha24": data["alpha_y24"][evaluate].ravel(),
                          "market_pred4": np.repeat(all_pred[4][0], 12),
                          "market_pred24": np.repeat(market_p24, 12),
                          "alpha_pred4_base": all_pred[4][1]["base"].ravel(),
                          "alpha_pred4_geometry": all_pred[4][1]["geometry"].ravel(),
                          "alpha_pred24_base": alpha_p24["base"].ravel(),
                          "alpha_pred24_geometry": alpha_p24["geometry"].ravel(),
                          "alpha_pred24_full_x2": alpha_p24["full_x2"].ravel(),
                          "alpha_pred24_full_x2_own": alpha_p24["full_x2_own"].ravel(),
                          "high_vol_p": risk.ravel(),
                          "high_vol_actual": data["high_vol_actual"][evaluate].ravel(),
                          "nearest_prototype": data["prototype"][evaluate].ravel(),
                          "prototype_distance": data["distance"][evaluate].ravel(),
                          "prototype_margin": data["margin"][evaluate].ravel()})
    for horizon in (4, 24):
        probability = data[f"return_p{horizon}"][evaluate].reshape(-1, 5)
        for bucket in range(5):
            frame[f"return{horizon}_bin_p{bucket}"] = probability[:, bucket]
        scale = np.sqrt(np.maximum(data["past_rv"][evaluate] /
                                   (6 if horizon == 4 else 1), 1e-10))
        frame[f"return{horizon}_bin_actual"] = np.digitize(
            data[f"return{horizon}"][evaluate] / scale,
            CFG["return_bin_edges"]).ravel()
    regime = np.where(data["market_30d"][evaluate] >= 0, "past30_up", "past30_down")
    regime = np.char.add(regime, np.where(data["market_risk"][evaluate] >= data["risk_cut"],
                                          "_high_risk", "_low_risk"))
    frame["past_market_regime"] = np.repeat(regime, 12)
    for item in ledger_rows:
        item["past_market_regime"] = regime
    threshold_row = {"phase": phase, "market_enter": market_enter, "market_exit": market_leave,
                     "pair_enter": pair_enter, "pair_exit": pair_leave,
                     "pair_geometry_enter": geom_enter, "pair_geometry_exit": geom_leave,
                     "pair_full_x2_enter": full_enter, "pair_full_x2_exit": full_leave,
                     "pair_full_x2_sparse_enter": float(np.quantile(pair_quantiles["full_x2"], .80)),
                     "pair_full_x2_sparse_exit": float(np.quantile(pair_quantiles["full_x2"], .40)),
                     "pair_full_x2_own_enter": own_enter, "pair_full_x2_own_exit": own_leave,
                     "pair_full_x2_own_sparse_enter": float(np.quantile(pair_quantiles["full_x2_own"], .80)),
                     "pair_full_x2_own_sparse_exit": float(np.quantile(pair_quantiles["full_x2_own"], .40)),
                     "market_momentum_enter": mom_market_enter,
                     "pair_momentum_enter": mom_pair_enter,
                     "fit_through": train_end, "first_decision": str(eval_time.iloc[0]),
                     "last_decision": str(eval_time.iloc[-1])}
    (OUT / f"thresholds_{phase}.json").write_text(json.dumps(threshold_row, indent=2), encoding="utf-8")
    frame.to_parquet(OUT / f"forecast_rows_{phase}.parquet", index=False, compression="zstd")
    sensitivity = []
    for name, pred, history in (("base", alpha_p24["base"], pair_quantiles["base"]),
                                ("geometry", alpha_p24["geometry"], pair_quantiles["geometry"]),
                                ("full_x2", alpha_p24["full_x2"], pair_quantiles["full_x2"]),
                                ("full_x2_own", alpha_p24["full_x2_own"],
                                 pair_quantiles["full_x2_own"])):
        for enter_q, exit_q in ((.60, .30), (.70, .35), (.80, .40)):
            entry, exit_ = np.quantile(history, [enter_q, exit_q])
            w = pair_weights(pred, beta, entry, exit_)
            trial = ledger(w, px, 4)
            sensitivity.append({"phase": phase, "model": name, "entry_quantile": enter_q,
                "exit_quantile": exit_q, "total_return_4bp": trial["net_wealth"][-1] - 1,
                "active_days": int((np.abs(w).sum(axis=1) > 0).sum()),
                "turnover": trial["turnover"].sum()})
    pd.DataFrame(sensitivity).to_csv(OUT / f"threshold_neighborhood_{phase}.csv", index=False)
    return audit, quantile_cards, pd.concat(ledger_rows), pd.concat(positions), pd.DataFrame(score_rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    length = len(panel) // len(SYMBOLS)
    times = panel.available_at.iloc[:length].reset_index(drop=True)
    for j in range(1, len(SYMBOLS)):
        if not panel.available_at.iloc[j * length:(j + 1) * length].reset_index(drop=True).equals(times):
            raise AssertionError("coin snapshot clocks differ")
    beta_all, momentum_all, market_30d, market_risk = read_past_beta(times)
    references = joblib.load(V2 / "frozen_feature_maps.joblib")
    rep = make_representation(panel, CFG, reference_maps=references)
    symbols = panel.symbol_code.to_numpy(dtype=np.int8)
    compact, names, _ = compact_features(rep["X2_ordered_path"], symbols,
                                         panel.rv_slow_15m.to_numpy())
    daily = np.flatnonzero((times.dt.hour == 0).to_numpy() & (times.dt.minute == 0).to_numpy())
    daily = daily[(daily >= 96 * 30) & (daily < length - 96)]
    good = np.isfinite(panel.return24.to_numpy().reshape(12, length)[:, daily]).all(axis=0)
    daily = daily[good]
    n = len(daily)
    rows = np.concatenate([daily + j * length for j in range(12)])
    x = compact[rows]
    market_x = np.column_stack((x[:n, 18:28], x[:, 28].reshape(12, n).mean(axis=0))).astype(np.float32)
    coin_x = np.column_stack((x, np.eye(12, dtype=np.float32)[symbols[rows]])).astype(np.float32)
    coin_full_x = np.column_stack((rep["X2_ordered_path"][rows],
                                   np.tile(market_x, (12, 1)),
                                   np.eye(12, dtype=np.float32)[symbols[rows]])).astype(np.float32)
    coin_full_own_x = np.column_stack((rep["X2_ordered_path"][rows],
                                       np.eye(12, dtype=np.float32)[symbols[rows]])).astype(np.float32)
    resolution = joblib.load(V2 / "resolution_chosen_model.joblib")
    nearest, distance, _, margin = center_membership(rep["X1_endpoints"][rows, :-4],
                resolution["centers"], resolution["temperature"])
    coin_geom_x = np.column_stack((coin_x, distance, margin)).astype(np.float32)
    distribution_models = joblib.load(V2 / "hybrid_compact_with_geometry_margin.joblib")["models"]
    return_p4 = distribution_models[0].predict_proba(coin_geom_x).reshape(12, n, 5).transpose(1, 0, 2)
    return_p24 = distribution_models[1].predict_proba(coin_geom_x).reshape(12, n, 5).transpose(1, 0, 2)
    high_vol_p = distribution_models[2].predict_proba(coin_geom_x)[:, 1].reshape(12, n).T
    rv4 = panel.future_rv4.to_numpy().reshape(12, length)[:, daily].T
    past_rv = panel.rv_slow_15m.to_numpy().reshape(12, length)[:, daily].T
    data = {"time": times.iloc[daily].reset_index(drop=True), "market_x": market_x,
            "coin_x": coin_x, "coin_geom_x": coin_geom_x, "coin_full_x": coin_full_x,
            "coin_full_own_x": coin_full_own_x,
            "beta": beta_all[daily], "momentum": momentum_all[daily],
            "market_30d": market_30d[daily], "market_risk": market_risk[daily],
            "risk_cut": float(np.nanmedian(market_risk[times < pd.Timestamp("2025-01-01", tz="UTC")])),
            "risk_p": high_vol_p, "high_vol_actual": (rv4 > past_rv / 6).astype(np.int8),
            "past_rv": past_rv,
            "return_p4": return_p4, "return_p24": return_p24,
            "prototype": nearest.reshape(12, n).T,
            "distance": distance.reshape(12, n).T,
            "margin": margin.reshape(12, n).T}
    for horizon in (4, 24):
        ret = panel[f"return{horizon}"].to_numpy().reshape(12, length)[:, daily].T
        market = ret.mean(axis=1)
        data[f"return{horizon}"] = ret
        data[f"market_y{horizon}"] = market
        data[f"alpha_y{horizon}"] = ret - data["beta"] * market[:, None]
    audits, quantiles, ledgers, holdings, scores = [], [], [], [], []
    for args in (("2025-07-01", "2025-07-01", "2026-01-01", "2025H2_development"),
                 ("2026-01-01", "2026-01-01", "2026-09-24", "2026_exploratory")):
        result = run_stage(data, *args)
        audit, quantile, book, position, score = result
        audits += audit
        quantiles += quantile
        ledgers.append(book)
        holdings.append(position)
        scores.append(score)
        print(args[-1], score.loc[score.fee_bp_side == 4,
              ["strategy", "total_return", "open_events", "avg_hold_hours"]].to_string(index=False), flush=True)
    audit = pd.DataFrame(audits)
    quantile = pd.concat(quantiles, ignore_index=True)
    book = pd.concat(ledgers, ignore_index=True)
    position = pd.concat(holdings, ignore_index=True)
    score = pd.concat(scores, ignore_index=True)
    audit.to_csv(OUT / "directional_audit.csv", index=False)
    quantile.to_csv(OUT / "conditional_return_quantiles.csv", index=False)
    book.to_parquet(OUT / "portfolio_daily.parquet", index=False, compression="zstd")
    position.to_parquet(OUT / "coin_positions_daily.parquet", index=False, compression="zstd")
    score.to_csv(OUT / "strategy_family_scores.csv", index=False)
    position["month"] = position.available_at.dt.strftime("%Y-%m")
    position.groupby(["phase", "strategy", "symbol", "month"]).agg(
        pnl_contribution=("asset_contribution", "sum"),
        active_days=("weight", lambda x: int((x != 0).sum())),
        long_days=("weight", lambda x: int((x > 0).sum())),
        short_days=("weight", lambda x: int((x < 0).sum()))).reset_index().to_csv(
            OUT / "coin_monthly_contributions.csv", index=False)
    uncertainty = []
    for phase in book.phase.unique():
        view = book[(book.phase == phase) & (book.fee_bp_side == 4)]
        pivot = view.pivot(index="available_at", columns="strategy", values="net_return")
        benchmark_daily = view[view.strategy == "pair_full_x2_direct"].buy_hold_return.to_numpy()
        full = pivot["pair_full_x2_direct"].to_numpy()
        for comparator, other in (("zero", np.zeros_like(full)),
                                  ("buy_hold", benchmark_daily),
                                  ("pair_base", pivot["pair_base"].to_numpy()),
                                  ("pair_k48_distance", pivot["pair_k48_distance"].to_numpy()),
                                  ("pair_momentum_same_entry", pivot["pair_momentum_same_entry"].to_numpy()),
                                  ("pair_reverse_full", pivot["pair_reverse_full"].to_numpy())):
            delta = full - other
            low, high = block_interval(delta)
            uncertainty.append({"phase": phase, "candidate": "pair_full_x2_direct",
                "comparator": comparator, "mean_delta_bp_day": 1e4 * delta.mean(),
                "week_block_low_bp_day": 1e4 * low, "week_block_high_bp_day": 1e4 * high})
        sparse = pivot["pair_full_x2_sparse_posthoc"].to_numpy()
        for comparator, other in (("zero", np.zeros_like(sparse)),
                                  ("buy_hold", benchmark_daily),
                                  ("pair_base", pivot["pair_base"].to_numpy()),
                                  ("pair_full_x2_direct", full),
                                  ("pair_momentum_sparse_same_entry",
                                   pivot["pair_momentum_sparse_same_entry"].to_numpy()),
                                  ("pair_reverse_sparse", pivot["pair_reverse_sparse"].to_numpy())):
            delta = sparse - other
            low, high = block_interval(delta)
            uncertainty.append({"phase": phase, "candidate": "pair_full_x2_sparse_posthoc",
                "comparator": comparator, "mean_delta_bp_day": 1e4 * delta.mean(),
                "week_block_low_bp_day": 1e4 * low, "week_block_high_bp_day": 1e4 * high})
        own = pivot["pair_full_x2_own_sparse_posthoc"].to_numpy()
        for comparator, other in (("zero", np.zeros_like(own)),
                                  ("buy_hold", benchmark_daily),
                                  ("pair_base", pivot["pair_base"].to_numpy()),
                                  ("pair_full_x2_sparse_posthoc", sparse),
                                  ("pair_momentum_own_sparse_same_entry",
                                   pivot["pair_momentum_own_sparse_same_entry"].to_numpy()),
                                  ("pair_reverse_own_sparse", pivot["pair_reverse_own_sparse"].to_numpy())):
            delta = own - other
            low, high = block_interval(delta)
            uncertainty.append({"phase": phase, "candidate": "pair_full_x2_own_sparse_posthoc",
                "comparator": comparator, "mean_delta_bp_day": 1e4 * delta.mean(),
                "week_block_low_bp_day": 1e4 * low, "week_block_high_bp_day": 1e4 * high})
    pd.DataFrame(uncertainty).to_csv(OUT / "strategy_paired_uncertainty.csv", index=False)
    book["month"] = book.available_at.dt.strftime("%Y-%m")
    monthly = book.groupby(["phase", "strategy", "fee_bp_side", "month"]).agg(
        net_return=("net_return", lambda x: np.prod(1 + x) - 1),
        buy_hold_return=("buy_hold_return", lambda x: np.prod(1 + x) - 1),
        active_days=("gross_exposure", lambda x: int((x > 0).sum())),
        market_component=("market_component", "sum"),
        residual_component=("residual_component", "sum")).reset_index()
    monthly.to_csv(OUT / "monthly_portfolio.csv", index=False)
    regime_card = book[book.fee_bp_side == 4].groupby(
        ["phase", "strategy", "past_market_regime"]).agg(
        days=("net_return", "size"), active_fraction=("gross_exposure", lambda x: (x > 0).mean()),
        mean_net_bp=("net_return", lambda x: 1e4 * x.mean()),
        market_component=("market_component", "sum"),
        residual_component=("residual_component", "sum")).reset_index()
    regime_card.to_csv(OUT / "market_regime_attribution.csv", index=False)
    cards = []
    for phase in ("2025H2_development", "2026_exploratory"):
        forecast = pd.read_parquet(OUT / f"forecast_rows_{phase}.parquet")
        forecast["month"] = forecast.available_at.dt.to_period("M").astype(str)
        state = forecast.groupby("nearest_prototype").agg(
            count=("return24", "size"), high_vol_rate=("high_vol_actual", "mean"),
            mean_return24=("return24", "mean"), mean_alpha24=("alpha24", "mean"),
            mean_high_vol_p=("high_vol_p", "mean"), mean_distance=("prototype_distance", "mean")).reset_index()
        state["phase"] = phase
        cards.append(state)
        forecast.groupby(["month", "symbol"])[["return24", "alpha24", "alpha_pred24_base",
                                                "high_vol_actual", "high_vol_p"]].mean().to_csv(
            OUT / f"monthly_coin_state_{phase}.csv")
    pd.concat(cards).to_csv(OUT / "prototype_state_cards.csv", index=False)
    latest_forecast = pd.read_parquet(OUT / "forecast_rows_2026_exploratory.parquet")
    latest_at = latest_forecast.available_at.max()
    latest_forecast = latest_forecast[latest_forecast.available_at == latest_at]
    latest_position = position[(position.phase == "2026_exploratory") &
                               (position.strategy == "pair_full_x2_own_sparse_posthoc") &
                               (position.available_at == latest_at)][["symbol", "weight"]]
    latest_forecast.merge(latest_position, on="symbol").to_csv(
        OUT / "latest_historical_state_snapshot.csv", index=False)
    reliability = []
    distribution_quality = []
    for phase in ("2025H2_development", "2026_exploratory"):
        fc = pd.read_parquet(OUT / f"forecast_rows_{phase}.parquet")
        for horizon in (4, 24):
            p = fc[[f"return{horizon}_bin_p{j}" for j in range(5)]].to_numpy()
            y = fc[f"return{horizon}_bin_actual"].to_numpy(dtype=int)
            loss = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))
            rps = np.mean((np.cumsum(p, axis=1)[:, :-1] -
                           (y[:, None] <= np.arange(4))) ** 2, axis=1)
            distribution_quality.append({"phase": phase, "horizon_h": horizon,
                "rows": len(y), "logloss": float(loss.mean()), "rps": float(rps.mean()),
                "negative_tail_forecast": float(p[:, 0].mean()),
                "negative_tail_actual": float((y == 0).mean()),
                "positive_tail_forecast": float(p[:, 4].mean()),
                "positive_tail_actual": float((y == 4).mean())})
        fc["probability_decile"] = pd.qcut(fc.high_vol_p.rank(method="first"), 10, labels=False)
        card = fc.groupby("probability_decile").agg(
            rows=("high_vol_actual", "size"), forecast_mean=("high_vol_p", "mean"),
            actual_rate=("high_vol_actual", "mean")).reset_index()
        card["phase"] = phase
        card["brier"] = float(np.mean((fc.high_vol_p - fc.high_vol_actual) ** 2))
        reliability.append(card)
    reliability = pd.concat(reliability)
    reliability.to_csv(OUT / "high_vol_reliability.csv", index=False)
    pd.DataFrame(distribution_quality).to_csv(OUT / "return_distribution_quality_daily.csv", index=False)
    plot_results(book, position, quantile, reliability)
    (OUT / "run_manifest.json").write_text(json.dumps({"symbols": SYMBOLS,
        "decision": "daily UTC 00:00 completed 15m snapshot; 00:05 next 5m execution open",
        "feature_reference": "pre-2025 v2 per-symbol empirical maps",
        "geometry": "v2 X1 clock-free K48 fixed centers, distance/margin only",
        "full_x2_direct": "100 X2 coordinates plus 11 completed-market/context fields and 12 symbol dummies",
        "full_x2_own_direct": "100 X2 coordinates plus 12 symbol dummies, without added market summary",
        "risk_probability": "frozen v2 compact plus geometry 4h high-vol classifier",
        "return_distribution": "frozen v2 compact plus geometry 4h/24h five-bin classifiers; separate from continuous-mean heads",
        "train": "2023-02 through 2025-06; refit to 2025-12 with 24h label purge",
        "2026_status": "exploratory reuse after v1/v2 inspection",
        "posthoc_candidates": ["full_x2_direct added after first compact strategy run",
                               "full_x2_sparse_80_40 chosen after threshold-neighborhood inspection",
                               "full_x2 own-only versus market-augmented ablation after 2026 outcome inspection"],
        "cost": "0, 4, 10 bp per side scenarios; excludes funding, spread and impact",
        "frequency": "one daily decision and 24h holding/review; 4h forecast audited only"},
        indent=2, ensure_ascii=False), encoding="utf-8")


def plot_results(book: pd.DataFrame, position: pd.DataFrame, quantile: pd.DataFrame,
                 reliability: pd.DataFrame) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, phase in zip(axs, book.phase.unique()):
        focus = book[(book.phase == phase) & (book.fee_bp_side == 4)]
        for name, label in (("pair_full_x2_own_sparse_posthoc", "own X2 sparse (post hoc)"),
                            ("pair_base", "compact state"),
                            ("pair_k48_distance", "compact + K48 distance")):
            part = focus[focus.strategy == name]
            ax.plot(part.available_at, part.net_wealth, label=label)
        part = focus[focus.strategy == "pair_base"]
        ax.plot(part.available_at, part.buy_hold_wealth, color="black", label="buy and hold")
        ax.axhline(1, color="gray", linewidth=.7)
        ax.set_title(phase)
        ax.tick_params(axis="x", rotation=25, labelsize=7)
        ax.legend(fontsize=7)
    axs[0].set_ylabel("wealth / initial capital; 4 bp per side scenario")
    fig.tight_layout()
    fig.savefig(OUT / "focused_strategy_comparison.png", dpi=160)
    plt.close(fig)
    for phase in book.phase.unique():
        view = book[(book.phase == phase) & (book.fee_bp_side == 4)]
        fig, ax = plt.subplots(figsize=(12, 5))
        for name in ("market_base", "market_risk_gate", "pair_base",
                     "pair_k48_distance", "pair_full_x2_direct",
                     "pair_full_x2_sparse_posthoc", "pair_full_x2_own_sparse_posthoc"):
            sub = view[view.strategy == name]
            ax.plot(sub.available_at, sub.net_wealth, label=name)
        base = view[view.strategy == "market_base"]
        ax.plot(base.available_at, base.buy_hold_wealth, color="black", linewidth=2,
                label="equal-capital buy and hold")
        ax.axhline(1, color="gray", linewidth=.7)
        ax.set_title(f"{phase}: daily executable wealth, 4 bp per side scenario")
        ax.set_ylabel("wealth / initial capital")
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / f"wealth_{phase}.png", dpi=160)
        plt.close(fig)
        pos = position[(position.phase == phase) &
                       (position.strategy == "pair_full_x2_own_sparse_posthoc")]
        heat = pos.pivot(index="symbol", columns="available_at", values="weight").loc[SYMBOLS]
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.imshow(heat.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-.6, vmax=.6)
        ax.set_yticks(range(12), SYMBOLS, fontsize=7)
        ax.set_title(f"{phase}: sparse own-X2 candidate per-coin weight (red long, blue short)")
        ax.set_xlabel("daily decisions")
        fig.tight_layout()
        fig.savefig(OUT / f"coin_weights_{phase}.png", dpi=160)
        plt.close(fig)
        pair = pos.copy()
        pair["cumulative_contribution"] = pair.groupby("symbol").asset_contribution.cumsum()
        fig, axs = plt.subplots(3, 4, figsize=(15, 8), sharex=True)
        for ax, symbol in zip(axs.ravel(), SYMBOLS):
            piece = pair[pair.symbol == symbol]
            ax.plot(piece.available_at, piece.cumulative_contribution, label="pair P&L contribution")
            ax.plot(piece.available_at, piece.coin_buy_hold_wealth - 1, alpha=.6,
                    label="coin buy and hold")
            ax.axhline(0, color="gray", linewidth=.5)
            ax.set_title(symbol, fontsize=9)
            ax.tick_params(axis="x", rotation=30, labelsize=6)
        axs[0, 0].legend(fontsize=6)
        fig.suptitle(f"{phase}: sparse own-X2 coin contribution vs individual buy and hold")
        fig.tight_layout()
        fig.savefig(OUT / f"coin_cumulative_{phase}.png", dpi=160)
        plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, phase in zip(axs, quantile.phase.unique()):
        q = quantile[(quantile.phase == phase) & (quantile.horizon_h == 24)]
        for name in q.model.unique():
            part = q[q.model == name]
            ax.plot(part["quantile"] + 1, part["mean"] * 1e4, marker="o", label=name)
        ax.axhline(0, color="gray", linewidth=.7)
        ax.set_title(phase)
        ax.set_xlabel("predicted residual quintile")
        ax.legend(fontsize=8)
    axs[0].set_ylabel("realized 24h residual, bp")
    fig.tight_layout()
    fig.savefig(OUT / "residual_forecast_quintiles.png", dpi=160)
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    for ax, phase in zip(axs, reliability.phase.unique()):
        card = reliability[reliability.phase == phase]
        ax.plot(card.forecast_mean, card.actual_rate, marker="o")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=.8)
        ax.set_title(f"{phase}; Brier={card.brier.iloc[0]:.3f}")
        ax.set_xlabel("predicted 4h high-vol probability")
    axs[0].set_ylabel("realized frequency")
    fig.tight_layout()
    fig.savefig(OUT / "high_vol_reliability.png", dpi=160)
    plt.close(fig)
    regime = pd.read_csv(OUT / "market_regime_attribution.csv")
    regime = regime[regime.strategy == "pair_full_x2_own_sparse_posthoc"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, phase in zip(axs, regime.phase.unique()):
        card = regime[regime.phase == phase]
        ax.bar(card.past_market_regime, card.mean_net_bp)
        ax.axhline(0, color="black", linewidth=.7)
        ax.set_title(f"{phase}: sparse candidate by known 30d regime")
        ax.tick_params(axis="x", rotation=35, labelsize=7)
    axs[0].set_ylabel("mean net bp per calendar day")
    fig.tight_layout()
    fig.savefig(OUT / "market_regime_attribution.png", dpi=160)
    plt.close(fig)
    cards = pd.read_csv(OUT / "prototype_state_cards.csv")
    a = cards[cards.phase == "2025H2_development"].set_index("nearest_prototype")
    b = cards[cards.phase == "2026_exploratory"].set_index("nearest_prototype")
    shared = a.index.intersection(b.index)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(a.loc[shared, "high_vol_rate"], b.loc[shared, "high_vol_rate"],
               s=np.sqrt(np.minimum(a.loc[shared, "count"], b.loc[shared, "count"])) * 5,
               alpha=.65)
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=.8)
    ax.set(xlabel="2025H2 realized high-vol rate", ylabel="2026 realized high-vol rate",
           title="K48 region outcome drift; point size reflects minimum support")
    fig.tight_layout()
    fig.savefig(OUT / "prototype_volatility_drift.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
