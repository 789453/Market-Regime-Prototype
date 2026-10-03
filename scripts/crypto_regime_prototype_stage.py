"""Completed-bar, forward-block research on regimes and predictive prototypes.

New artifacts live under reports/crypto/regime_prototypes_v1.  Original OHLCV
and all earlier models/reports are read only.  2026 is reused exploration.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.crypto.full_patterns import EmpiricalMap
from src.crypto.regime_prototypes import (REGIMES, ENTER_QUANTILES, EXIT_QUANTILES,
    adaptive_pair_weights, classify_regime, fit_predictive_prototypes,
    future_path_labels, select_beta_pair, semivariance_direction)
from src.crypto.signal_chain import ledger, buy_hold

CFG = yaml.safe_load((ROOT / "configs/crypto_regime_prototypes.yaml").read_text(encoding="utf-8"))
REP_CFG = yaml.safe_load((ROOT / "configs/crypto_predictive_states.yaml").read_text(encoding="utf-8"))
OUT = ROOT / "reports/crypto/regime_prototypes_v1"
SEED = CFG["random_seed"]
SYMBOLS = CFG["symbols"]
FIELDS = tuple(dict.fromkeys(n for group in REP_CFG["representation_columns"].values() for n in group))


def read_symbol(symbol: str) -> dict:
    source = Path(CFG["source"]) / symbol
    feature_path = ROOT / CFG["feature_dir"] / f"{symbol}.parquet"
    cols = ["available_at", *FIELDS, "rv_slow_15m"]
    frame = pq.read_table(feature_path, columns=list(dict.fromkeys(cols)),
                          filters=[("available_at", "<", pd.Timestamp(CFG["source_end_exclusive"]))]).to_pandas()
    times_all = pd.DatetimeIndex(frame.available_at)
    fit = frame.loc[times_all < pd.Timestamp(CFG["fit_reference_end"])]
    mapper = EmpiricalMap(fit, FIELDS, sample=25000)
    mapped, missing = mapper.transform(frame)
    idx = np.flatnonzero((times_all.minute == 0) &
                         (times_all >= pd.Timestamp(CFG["training_start"])) &
                         (times_all <= pd.Timestamp(CFG["fold_edges"][-1])))
    if np.min(idx) < 16:
        raise AssertionError("path history unavailable")
    blocks = []
    geometry = []
    offset = 0
    for group, names in REP_CFG["representation_columns"].items():
        current = mapped[idx, offset:offset + 4]
        if group in REP_CFG["path_columns"]:
            past = [mapped[idx - lag, offset + 2:offset + 4] for lag in (16, 12, 8, 4)]
            nodes = np.stack((past[0], past[1], past[2], past[3], current[:, 2:4]), axis=1)
            block = np.concatenate((current, *past, nodes.argmax(axis=1) / 4,
                                    nodes.argmin(axis=1) / 4), axis=1)
            geometry.extend((current.mean(axis=1),
                             (current[:, 2:4] - past[3]).mean(axis=1),
                             (past[3] - past[0]).mean(axis=1)))
        else:
            block = current
        blocks.append((2 * block - 1) / np.sqrt(block.shape[1]))
        offset += 4
    x2 = np.concatenate(blocks, axis=1).astype(np.float32)
    geo = np.stack(geometry, axis=1).astype(np.float32)
    if x2.shape[1] != 100 or geo.shape[1] != 18:
        raise AssertionError("unexpected representation width")
    rv = np.maximum(frame.rv_slow_15m.to_numpy(dtype=np.float64)[idx], 1e-10)

    hour = pq.read_table(source / "1h.parquet", columns=["open_time", "close"]).to_pandas()
    hour_available = (pd.DatetimeIndex(pd.to_datetime(hour.open_time, unit="ms", utc=True))
                      + pd.Timedelta(hours=1)).as_unit("ns")
    hpos = np.searchsorted(hour_available.asi8, times_all[idx].asi8)
    if not np.array_equal(hour_available.asi8[hpos], times_all[idx].asi8):
        raise AssertionError(f"1h completed bar mismatch: {symbol}")
    logclose = np.log(hour.close.to_numpy(dtype=np.float64)[hpos])

    raw = pq.read_table(source / "5m.parquet", columns=["open_time", "open"]).to_pandas()
    stamp = raw.open_time.to_numpy(dtype=np.int64)
    logopen = np.log(raw.open.to_numpy(dtype=np.float64))
    request = times_all[idx].asi8 // 1_000_000 + 300_000
    entry = np.searchsorted(stamp, request)
    if np.any(entry >= len(stamp)) or not np.array_equal(stamp[entry], request):
        raise AssertionError(f"missing 00:05 style execution price: {symbol}")
    path = future_path_labels(logopen, entry)
    # The last fold ends five days before the raw history, so every horizon
    # must be available for all modeled observations.
    if not all(np.isfinite(values).all() for values in path.values()):
        raise AssertionError(f"incomplete future observation: {symbol}")
    # Future MFE/MAE is a training/evaluation label only.
    future = logopen[entry[:, None] + np.arange(1, 24 * 12 + 1)[None, :]]
    mfe = future.max(axis=1) - logopen[entry]
    mae = future.min(axis=1) - logopen[entry]
    return {"time": times_all[idx], "x": x2, "geo": geo, "rv": rv,
            "close": logclose, "price": logopen[entry], "missing": missing[idx].sum(axis=1),
            "mfe24": mfe, "mae24": mae, **path}


def load_data() -> dict:
    pieces = []
    for symbol in SYMBOLS:
        pieces.append(read_symbol(symbol))
        print(f"loaded {symbol}", flush=True)
    time = pieces[0]["time"]
    if not all(np.array_equal(time.asi8, p["time"].asi8) for p in pieces[1:]):
        raise AssertionError("cross-coin clock mismatch")
    data = {"time": time}
    for key in pieces[0]:
        if key != "time":
            data[key] = np.stack([p[key] for p in pieces], axis=1)
    close = data["close"]
    hourly = np.diff(close, axis=0, prepend=close[:1])
    market = hourly.mean(axis=1)
    market_series = pd.Series(market)
    win = 24 * 30
    mavg = market_series.rolling(win, min_periods=24 * 7).mean().to_numpy()
    mvar = pd.Series(market ** 2).rolling(win, min_periods=24 * 7).mean().to_numpy() - mavg ** 2
    beta = np.empty_like(close)
    for j in range(len(SYMBOLS)):
        r = hourly[:, j]
        ravg = pd.Series(r).rolling(win, min_periods=24 * 7).mean().to_numpy()
        cross = pd.Series(r * market).rolling(win, min_periods=24 * 7).mean().to_numpy()
        beta[:, j] = np.clip((cross - ravg * mavg) / np.maximum(mvar, 1e-12), .2, 2.5)
    trend = np.full(len(close), np.nan)
    trend[win:] = (close[win:] - close[:-win]).mean(axis=1)
    data["beta"] = beta
    data["market_trend"] = trend
    data["market_rv"] = data["rv"].mean(axis=1)
    for h in (4, 24, 72, 120):
        data[f"alpha{h}"] = data[f"return{h}"] - beta * data[f"return{h}"].mean(axis=1)[:, None]
    scale4 = np.sqrt(np.maximum(data["rv"] / 6, 1e-9))
    scale24 = np.sqrt(np.maximum(data["rv"], 1e-9))
    def bounded(v):
        return np.clip(v, -5, 5)
    signature = np.stack((bounded(data["alpha4"] / scale4),
                          bounded(data["alpha24"] / scale24),
                          semivariance_direction(data["up4"], data["down4"]),
                          semivariance_direction(data["up24"], data["down24"]),
                          bounded(np.log((data["up4"] + data["down4"] + 1e-10) /
                                         (data["rv"] / 6 + 1e-10))),
                          bounded(np.log((data["up24"] + data["down24"] + 1e-10) /
                                         (data["rv"] + 1e-10))),
                          bounded(data["mfe24"] / scale24),
                          bounded(data["mae24"] / scale24)), axis=2).astype(np.float32)
    data["signature"] = signature
    data["scale24"] = scale24
    return data


def lgb_fit(x: np.ndarray, y: np.ndarray, seed: int,
            *, trees: int = 110, leaves: int = 10) -> lgb.LGBMRegressor:
    reg = lgb.LGBMRegressor(n_estimators=trees, num_leaves=leaves,
                           learning_rate=.045, min_child_samples=180,
                           reg_lambda=20, colsample_bytree=.85, n_jobs=6,
                           verbosity=-1, random_state=seed)
    reg.fit(x, np.clip(y, *np.quantile(y, [.005, .995])))
    return reg


def block_interval(values: np.ndarray, seed: int = SEED, block: int = 14) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    start = rng.integers(0, n, (1000, int(np.ceil(n / block))))
    draws = values[((start[:, :, None] + np.arange(block)) % n).reshape(1000, -1)[:, :n]].mean(axis=1)
    return tuple(np.quantile(draws, [.025, .975]))


def calibrate_thresholds(history: pd.DataFrame, current: pd.Timestamp,
                         forecast: str, regime: int, *, state_specific: bool) -> tuple[float, float]:
    past = history[(history.time < current) &
                   (history.time >= current - pd.Timedelta(days=CFG["forecast_lookback_days"])) &
                   (history.forecast == forecast)]
    if state_specific:
        local = past[past.regime == regime]
        if len(local) >= 60:
            past = local
    if len(past) < 100:
        # No future label is needed, but a history of actual forward forecasts
        # is required before this policy can issue a new position.
        return np.inf, np.inf
    state = REGIMES[regime]
    eq = ENTER_QUANTILES[state] if state_specific else .80
    xq = EXIT_QUANTILES[state] if state_specific else .40
    return tuple(np.quantile(past.score, [eq, xq]))


def make_policy_history(time: pd.DatetimeIndex, regime: np.ndarray,
                        beta: np.ndarray, forecasts: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, matrix in forecasts.items():
        scores = np.array([select_beta_pair(matrix[t], beta[t])[2] for t in range(len(time))])
        rows.append(pd.DataFrame({"time": time, "regime": regime, "forecast": name, "score": scores}))
    return pd.concat(rows, ignore_index=True)


def policy_threshold_arrays(time: pd.DatetimeIndex, regime: np.ndarray,
                            prior: pd.DataFrame, current_scores: pd.DataFrame,
                            forecast: str, *, state_specific: bool) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    ent = np.full(len(time), np.inf)
    ext = np.full(len(time), np.inf)
    cards = []
    months = time.to_period("M")
    for month in months.unique():
        index = np.flatnonzero(months == month)
        first = time[index[0]]
        eligible = pd.concat((prior, current_scores[current_scores.time < first]), ignore_index=True)
        for state in range(len(REGIMES)):
            e, x = calibrate_thresholds(eligible, first, forecast, state,
                                        state_specific=state_specific)
            chosen = index[regime[index] == state]
            ent[chosen], ext[chosen] = e, x
            cards.append({"month": str(month), "forecast": forecast,
                          "policy": "state" if state_specific else "uniform",
                          "regime": REGIMES[state], "enter": e, "exit": x,
                          "hours": len(chosen)})
    return ent, ext, pd.DataFrame(cards)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_data()
    time: pd.DatetimeIndex = data["time"]
    n, coins, width = data["x"].shape
    print(f"hourly panel {n:,} clocks x {coins} symbols; X2 width={width}", flush=True)
    valid = (np.isfinite(data["market_trend"]) & np.isfinite(data["beta"]).all(axis=1) &
             np.isfinite(data["signature"]).all(axis=(1, 2)))
    if not valid[time >= pd.Timestamp(CFG["fold_edges"][0])].all():
        raise AssertionError("nonfinite modeled sample")
    eye = np.eye(coins, dtype=np.float32)
    x_direct = np.concatenate((data["x"], np.broadcast_to(eye, (n, coins, coins))), axis=2)
    folds = [pd.Timestamp(s) for s in CFG["fold_edges"]]
    forecast_rows = []
    prototype_cards = []
    scores = []
    policies = []
    daily_rows = []
    position_rows = []
    threshold_rows = []
    fold_rows = []
    identity_rows = []
    old_proto = None
    history = pd.DataFrame(columns=["time", "regime", "forecast", "score"])
    for f in range(len(folds) - 1):
        start, end = folds[f:f + 2]
        train_time = ((time >= pd.Timestamp(CFG["training_start"])) &
                      (time + pd.Timedelta(hours=CFG["longest_label_purge_hours"]) < start) & valid)
        eval_time = (time >= start) & (time < end) & valid
        if train_time.sum() < 1000 or eval_time.sum() < 100:
            raise AssertionError(f"invalid fold {start}")
        tr = np.flatnonzero(train_time)
        ev = np.flatnonzero(eval_time)
        qlo, qhi = np.quantile(data["market_rv"][tr], [.33, .67])
        regime = classify_regime(data["market_rv"][ev], data["market_trend"][ev], qlo, qhi)
        xt = x_direct[tr].reshape(-1, x_direct.shape[2])
        xe = x_direct[ev].reshape(-1, x_direct.shape[2])
        yt = data["signature"][tr].reshape(-1, 8)
        print(f"fold {f}: {start.date()} to {end.date()} train {len(xt):,} eval {len(xe):,}", flush=True)
        direct_heads = {}
        for task in (0, 1, 3, 5):
            direct_heads[task] = lgb_fit(xt, yt[:, task], SEED + f * 11 + task)
        direct = {task: model.predict(xe).reshape(len(ev), coins) for task, model in direct_heads.items()}
        calm_train = np.repeat(data["market_rv"][tr] < qlo, coins)
        calm_model = lgb_fit(xt[calm_train], yt[calm_train, 1], SEED + f * 11 + 9)
        calm24 = calm_model.predict(xe).reshape(len(ev), coins)

        mean = yt.mean(axis=0)
        std = np.maximum(yt.std(axis=0), .03)
        standardized = (yt - mean) / std
        x_proto = data["x"][tr].reshape(-1, width)
        x_proto_eval = data["x"][ev].reshape(-1, width)
        same_input_direct = {}
        for task in (1, 3, 5):
            same_input_direct[task] = lgb_fit(x_proto, yt[:, task],
                                              SEED + f * 11 + 30 + task).predict(x_proto_eval).reshape(len(ev), coins)
        geo_proto = data["geo"][tr].reshape(-1, 18)
        dates = np.repeat(time[tr].floor("D").asi8, coins)
        prototype = fit_predictive_prototypes(x_proto, geo_proto, standardized, dates,
                                               clusters=CFG["prototype_parents"], seed=SEED + f)
        predicted_z, parent, leaf = prototype.predict(x_proto_eval,
                                                     data["geo"][ev].reshape(-1, 18))
        proto = (predicted_z * std + mean).reshape(len(ev), coins, 8)
        raw_parent = (prototype.parent_means[parent] * std + mean).reshape(len(ev), coins, 8)
        # Distinct claim: a child is a direction prototype only if its later
        # training block improves the two directional path targets together.
        direction = fit_predictive_prototypes(x_proto, geo_proto, standardized, dates,
            clusters=CFG["prototype_parents"], seed=SEED + f,
            fit_task_weights=np.array([.2, 2., .2, 2., .1, .1, .1, .1]),
            certify_tasks=(1, 3))
        directional_z, direction_parent, direction_leaf = direction.predict(
            x_proto_eval, data["geo"][ev].reshape(-1, 18))
        directional = (directional_z * std + mean).reshape(len(ev), coins, 8)
        if old_proto is not None:
            anchor = data["geo"][tr[::max(1, len(tr) // 3000)]].reshape(-1, 18)
            identity_rows.append({"fold": f, "start": str(start),
                                  "parent_ari_common_history": adjusted_rand_score(
                                      old_proto.geometry.predict(anchor), prototype.geometry.predict(anchor))})
        old_proto = prototype
        for k in range(CFG["prototype_parents"]):
            candidate = np.flatnonzero(prototype.geometry.predict(geo_proto[::20]) == k)
            if len(candidate):
                coarse = candidate[np.argmin(np.sum((geo_proto[::20][candidate] - prototype.geometry.cluster_centers_[k]) ** 2, axis=1))]
                sample_index = int(coarse * 20)
            else:
                sample_index = 0
            proto_time = tr[sample_index // coins]
            proto_symbol = SYMBOLS[sample_index % coins]
            prototype_cards.append({"fold": f, "start": str(start), "parent": k,
                                    "prototype_type": "multitask_risk",
                                    "historical_rows": int(prototype.train_counts[k]),
                                    "supervised_split_accepted": bool(prototype.accepted[k]),
                                    "split_feature_x2": int(prototype.split_features[k]),
                                    "medoid_time": str(time[proto_time]), "medoid_symbol": proto_symbol,
                                    **{f"condition_{i}": float((prototype.parent_means[k] * std + mean)[i]) for i in range(8)}})
            prototype_cards.append({"fold": f, "start": str(start), "parent": k,
                                    "prototype_type": "direction_certified",
                                    "historical_rows": int(direction.train_counts[k]),
                                    "supervised_split_accepted": bool(direction.accepted[k]),
                                    "split_feature_x2": int(direction.split_features[k]),
                                    "medoid_time": str(time[proto_time]), "medoid_symbol": proto_symbol,
                                    **{f"condition_{i}": float((direction.parent_means[k] * std + mean)[i]) for i in range(8)}})
        scale = data["scale24"][ev]
        own = direct[1] * scale
        proto_alpha = proto[:, :, 1] * scale
        calm_alpha = calm24 * scale
        direction_alpha = directional[:, :, 1] * scale
        calm_specialist = np.where(np.isin(regime, [0, 1])[:, None], calm_alpha, own)
        direction_hybrid = np.where(np.isin(regime, [0, 1])[:, None],
                                    .5 * calm_alpha + .5 * direction_alpha, own)
        hybrid = np.where(np.isin(regime, [0, 1])[:, None],
                          .5 * calm_alpha + .5 * proto_alpha,
                          np.where((regime == 3)[:, None], .8 * own + .2 * proto_alpha,
                                   .7 * own + .3 * proto_alpha))
        forecasts = {"direct": own, "prototype": proto_alpha, "hybrid": hybrid,
                     "directional": direction_alpha, "calm_specialist": calm_specialist,
                     "direction_hybrid": direction_hybrid}
        current_history = make_policy_history(time[ev], regime, data["beta"][ev], forecasts)

        # Proper forecast comparison: background, geometric parent, accepted
        # supervised children, and same-X2 direct learner, all on same rows.
        for state in range(len(REGIMES)):
            mask = regime == state
            if not mask.any():
                continue
            actual = data["signature"][ev][mask]
            background = np.broadcast_to(mean, actual.shape)
            for name, prediction in (("background", background),
                                     ("geometry", raw_parent[mask]),
                                     ("supervised_prototype", proto[mask]),
                                     ("direction_prototype", directional[mask])):
                for task in range(8):
                    scores.append({"fold": f, "start": str(start), "regime": REGIMES[state],
                                   "model": name, "task": task, "hours": int(mask.sum()),
                                   "mse": float(np.mean((actual[:, :, task] - prediction[:, :, task]) ** 2))})
            for task in direct:
                scores.append({"fold": f, "start": str(start), "regime": REGIMES[state],
                               "model": "direct_x2_symbol", "task": task, "hours": int(mask.sum()),
                               "mse": float(np.mean((actual[:, :, task] - direct[task][mask]) ** 2))})
            for task in same_input_direct:
                scores.append({"fold": f, "start": str(start), "regime": REGIMES[state],
                               "model": "direct_x2_only", "task": task, "hours": int(mask.sum()),
                               "mse": float(np.mean((actual[:, :, task] - same_input_direct[task][mask]) ** 2))})
        fold_rows.append({"fold": f, "start": str(start), "end": str(end),
                          "train_clocks": len(tr), "eval_clocks": len(ev),
                          "vol_cut_low": float(qlo), "vol_cut_high": float(qhi),
                          "supervised_children": int(prototype.accepted.sum()),
                          "direction_children": int(direction.accepted.sum()),
                          "missing_coordinates_eval": int(data["missing"][ev].sum())})
        columns = {"time": np.repeat(time[ev].to_numpy(), coins),
                   "symbol": np.tile(SYMBOLS, len(ev)),
                   "regime": np.repeat([REGIMES[v] for v in regime], coins),
                   "fold": f, "parent": parent, "leaf": leaf,
                   "direction_parent": direction_parent, "direction_leaf": direction_leaf,
                   "beta": data["beta"][ev].reshape(-1),
                   "future_alpha24": data["alpha24"][ev].reshape(-1),
                   "future_return24": data["return24"][ev].reshape(-1),
                   "future_side_balance4": data["signature"][ev, :, 2].reshape(-1),
                   "future_side_balance24": data["signature"][ev, :, 3].reshape(-1),
                   "future_log_rv_ratio24": data["signature"][ev, :, 5].reshape(-1),
                   "direct_alpha24": own.reshape(-1),
                   "same_input_direct_alpha24": (same_input_direct[1] * scale).reshape(-1),
                   "calm_alpha24": calm_specialist.reshape(-1),
                   "direction_alpha24": direction_alpha.reshape(-1),
                   "direction_hybrid_alpha24": direction_hybrid.reshape(-1),
                   "proto_alpha24": proto_alpha.reshape(-1),
                   "hybrid_alpha24": hybrid.reshape(-1),
                   "pred_side_balance24": proto[:, :, 3].reshape(-1),
                   "pred_log_rv_ratio24": proto[:, :, 5].reshape(-1),
                   "pred_mfe24": proto[:, :, 6].reshape(-1),
                   "pred_mae24": proto[:, :, 7].reshape(-1)}
        forecast_rows.append(pd.DataFrame(columns))
        if f == 0:
            history = pd.concat((history, current_history), ignore_index=True)
            continue
        for name in forecasts:
            for version in (["uniform", "state"] if name in ("direct", "hybrid") else ["state"]):
                state_specific = version == "state"
                e, x, cards = policy_threshold_arrays(time[ev], regime, history, current_history,
                                                      name, state_specific=state_specific)
                threshold_rows.append(cards)
                w, episodes = adaptive_pair_weights(forecasts[name], data["beta"][ev],
                                                    regime, e, x, continuous=state_specific)
                policy_name = f"{name}_{version}"
                prices = data["price"][np.r_[ev, ev[-1] + 1]]
                bench = buy_hold(prices)
                for fee in CFG["commission_scenarios_bp_side"]:
                    book = ledger(w, prices, fee)
                    policies.append({"fold": f, "start": str(start), "policy": policy_name,
                                     "fee_bp_side": fee, "hours": len(ev),
                                     "total_return": float(book["net_wealth"][-1] - 1),
                                     "buy_hold_total_return": float(bench[-1] - 1),
                                     "active_fraction": float((np.abs(w).sum(axis=1) > .05).mean()),
                                     "mean_gross": float(np.abs(w).sum(axis=1).mean()),
                                     "turnover": float(book["turnover"].sum()),
                                     "mean_net_bp_hour": float(book["net_return"].mean() * 1e4)})
                    if fee == 4:
                        daily_rows.append(pd.DataFrame({"time": time[ev], "fold": f,
                            "policy": policy_name, "regime": [REGIMES[v] for v in regime],
                            "net_return": book["net_return"], "gross_return": book["gross_return"],
                            "turnover": book["turnover"], "gross": np.abs(w).sum(axis=1),
                            "buy_hold_return": np.diff(bench) / bench[:-1]}))
                        position_rows.append(pd.DataFrame({"time": time[ev], "fold": f,
                            "policy": policy_name, "long": episodes[:, 0],
                            "short": episodes[:, 1], "age_hours": episodes[:, 2],
                            "gross": np.abs(w).sum(axis=1),
                            "weight_long": w.max(axis=1), "weight_short": w.min(axis=1)}))
        history = pd.concat((history, current_history), ignore_index=True)
        print(f"fold {f} completed: splits={prototype.accepted.sum()}", flush=True)

    prediction = pd.concat(forecast_rows, ignore_index=True)
    prediction.to_parquet(OUT / "forward_forecasts.parquet", index=False, compression="zstd")
    pd.DataFrame(scores).to_csv(OUT / "distribution_scores.csv", index=False)
    pd.DataFrame(prototype_cards).to_csv(OUT / "prototype_cards.csv", index=False)
    pd.DataFrame(identity_rows).to_csv(OUT / "identity_stability.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "fold_manifest.csv", index=False)
    pd.DataFrame(policies).to_csv(OUT / "policy_scores.csv", index=False)
    pd.concat(threshold_rows, ignore_index=True).to_csv(OUT / "threshold_history.csv", index=False)
    pd.concat(daily_rows, ignore_index=True).to_parquet(OUT / "hourly_policy_ledger.parquet", index=False, compression="zstd")
    pd.concat(position_rows, ignore_index=True).to_parquet(OUT / "policy_episodes.parquet", index=False, compression="zstd")
    (OUT / "run_manifest.json").write_text(json.dumps({"config": CFG, "symbols": SYMBOLS,
        "first_fold_role": "forward score calibration only",
        "2026_role": "reused exploration, not untouched test",
        "outcome_signature": ["4h beta residual / past sigma", "24h beta residual / past sigma",
            "4h up-minus-down semivariance balance", "24h up-minus-down semivariance balance",
            "4h log future/known variance", "24h log future/known variance",
            "24h maximum favorable excursion / past sigma", "24h maximum adverse excursion / past sigma"],
        "prototype_selection": "geometry parents; separate risk/multitask and direction-certified later-time split tests; refit accepted children on eligible train",
        "forecast_comparators": {"direct_x2_symbol": "100 X2 coordinates plus 12 symbol indicators",
                                 "direct_x2_only": "same 100 X2 coordinates used by prototype outcome split",
                                 "geometry": "18 direction-neutral X2 geometry coordinates"},
        "decision": "hourly completed bar at HH:00, next 5m open HH:05",
        "fee": "scenario, excludes funding/spread/impact"}, ensure_ascii=False, indent=2), encoding="utf-8")
    summarize(prediction, pd.DataFrame(scores), pd.DataFrame(policies),
              pd.concat(daily_rows, ignore_index=True), pd.concat(position_rows, ignore_index=True),
              pd.DataFrame(prototype_cards), pd.DataFrame(identity_rows))


def summarize(prediction: pd.DataFrame, score: pd.DataFrame, policy: pd.DataFrame,
              hourly: pd.DataFrame, episodes: pd.DataFrame,
              prototypes: pd.DataFrame, identity: pd.DataFrame) -> None:
    # Shared-time aggregation precedes uncertainty or annualized summaries.
    regime = hourly.groupby(["policy", "regime"]).agg(hours=("gross", "size"),
        active=("gross", lambda s: float((s > .05).mean())),
        avg_gross=("gross", "mean"), net_bp_hour=("net_return", lambda s: float(s.mean() * 1e4)),
        turnover=("turnover", "sum")).reset_index()
    regime.to_csv(OUT / "regime_policy_attribution.csv", index=False)
    distribution = score.groupby(["model", "regime", "task"]).apply(
        lambda g: np.average(g.mse, weights=g.hours), include_groups=False).rename("mse").reset_index()
    distribution.to_csv(OUT / "distribution_summary.csv", index=False)
    paired = []
    for state in REGIMES:
        a = score[(score.regime == state) & (score.model == "geometry")]
        for model in ("supervised_prototype", "direction_prototype"):
            b = score[(score.regime == state) & (score.model == model)]
            for task in range(8):
                x = a[a.task == task].sort_values("fold")
                y = b[b.task == task].sort_values("fold")
                if len(x) == len(y):
                    paired.append({"regime": state, "model": model, "task": task,
                                   "folds": len(x), "geometry_minus_supervised_mse":
                                   float(np.average(x.mse.to_numpy() - y.mse.to_numpy(), weights=x.hours.to_numpy()))})
    pd.DataFrame(paired).to_csv(OUT / "prototype_increment.csv", index=False)
    uncertainty = []
    for name, group in hourly.groupby("policy"):
        day = group.assign(day=group.time.dt.floor("D")).groupby("day").net_return.sum().to_numpy()
        lo, hi = block_interval(day)
        uncertainty.append({"policy": name, "daily_mean_bp": day.mean() * 1e4,
                            "block14_low_bp": lo * 1e4, "block14_high_bp": hi * 1e4,
                            "days": len(day)})
    pd.DataFrame(uncertainty).to_csv(OUT / "policy_uncertainty.csv", index=False)

    plt.rcParams.update({"figure.figsize": (11, 5), "axes.grid": True, "grid.alpha": .22})
    fig, ax = plt.subplots()
    for name, group in hourly.groupby("policy"):
        group = group.sort_values("time")
        ax.plot(group.time, np.cumprod(1 + group.net_return), label=name, linewidth=1.2)
    first = hourly[hourly.policy == hourly.policy.iloc[0]].sort_values("time")
    ax.plot(first.time, np.cumprod(1 + first.buy_hold_return), color="black", label="buy and hold", linewidth=1.4)
    ax.set(title="Forward development and reused 2026 exploration, 4 bp per side", ylabel="Wealth from 1")
    ax.legend(ncol=3, fontsize=8); fig.tight_layout(); fig.savefig(OUT / "01_wealth.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots()
    view = regime.pivot(index="regime", columns="policy", values="active").reindex(REGIMES)
    view.plot(kind="bar", ax=ax); ax.axhspan(.3, .6, alpha=.07, color="green")
    ax.set(title="Time participation by known market state", ylabel="Fraction with gross > 0.05", xlabel="")
    fig.tight_layout(); fig.savefig(OUT / "02_regime_coverage.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots()
    view = regime.pivot(index="regime", columns="policy", values="net_bp_hour").reindex(REGIMES)
    view.plot(kind="bar", ax=ax); ax.axhline(0, color="black", linewidth=.7)
    ax.set(title="Net return within each known state (4 bp scenario)", ylabel="bp per hour", xlabel="")
    fig.tight_layout(); fig.savefig(OUT / "03_regime_returns.png", dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    labels = {"direct_x2_only": "direct X2 only", "direct_x2_symbol": "direct X2 + symbol",
              "geometry": "geometry parent", "supervised_prototype": "risk prototype",
              "direction_prototype": "direction prototype"}
    for axis, task, title in zip(axes, (1, 3, 5),
                                 ("24h relative return", "24h up/down balance", "24h future risk")):
        view = distribution[distribution.task == task].pivot(
            index="regime", columns="model", values="mse").reindex(REGIMES)
        for model, label in labels.items():
            axis.plot(np.arange(len(REGIMES)), (view.background - view[model]) * 1000,
                      marker="o", linewidth=1.5, label=label)
        axis.axhline(0, color="black", linewidth=.8)
        axis.set_xticks(np.arange(len(REGIMES)), REGIMES, rotation=45, ha="right")
        axis.set(title=title, ylabel="Background MSE minus model MSE × 1000")
    handles, legends = axes[0].get_legend_handles_labels()
    fig.legend(handles, legends, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(.5, .01))
    fig.suptitle("Forward conditional prediction: positive is improvement over background")
    fig.tight_layout(rect=(0, .20, 1, .94)); fig.savefig(OUT / "04_prototype_forecast.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots()
    card = prototypes.pivot_table(index="start", columns="prototype_type",
                                  values="supervised_split_accepted", aggfunc="sum")
    card.plot(kind="bar", ax=ax)
    ax.set(title="Outcome splits retained after later-time validation", ylabel="Parents with retained split", xlabel="")
    fig.tight_layout(); fig.savefig(OUT / "05_predictive_splits.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots()
    subset = prediction[prediction.regime.isin(("calm_down", "calm_up"))]
    for label, field in (("direct", "direct_alpha24"), ("risk prototype", "proto_alpha24"),
                         ("direction prototype", "direction_alpha24"),
                         ("calm specialist", "calm_alpha24")):
        bucket = pd.qcut(subset[field].rank(method="first"), 8, labels=False)
        avg = subset.groupby(bucket).future_alpha24.mean() * 1e4
        ax.plot(np.arange(1, 9), avg, marker="o", label=label)
    ax.set(title="Calm-state predicted rank versus future 24h residual", xlabel="Prediction octile", ylabel="Realized bp")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "06_calm_rank.png", dpi=170); plt.close(fig)
    print("saved forecasts, policies, distributions and six figures", flush=True)


if __name__ == "__main__":
    main()
