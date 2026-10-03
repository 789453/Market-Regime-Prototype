"""Causal, bar-close feature definitions for 5m and 15m crypto bars."""

from __future__ import annotations

import numpy as np
import pandas as pd


GROUPS = {
    "vol_strength": ["rv_fast", "rv_med", "rv_slow", "rv_fast_slow", "rv_accel", "range_vol_fast", "body_vol_fast", "rv_surprise"],
    "vol_asymmetry": ["up_semivar_fast", "down_semivar_fast", "semivar_balance", "down_up_ratio", "negative_rate_med", "return_vol_corr_med", "upper_wick_med", "lower_wick_med"],
    "path_geometry": ["body_range", "close_location", "range_ratio", "open_gap_z", "vwap_distance", "above_prev_high", "below_prev_low", "reversal_from_high"],
    "trend": ["momentum_fast", "momentum_med", "momentum_slow", "momentum_z_med", "ema_fast_med", "ema_med_slow", "direction_alignment", "breakout_balance"],
    "efficiency": ["efficiency_fast", "efficiency_med", "sign_flip_med", "return_acf1_med", "variance_ratio_2", "choppiness_med", "sign_entropy_med", "path_range_ratio"],
    "participation": ["quote_rel_fast", "quote_rel_slow", "trades_rel_fast", "mean_trade_quote", "flow_imb_fast", "flow_imb_med", "flow_change", "impact_proxy"],
    "time_context": ["hour_sin", "hour_cos", "week_sin", "week_cos", "is_weekend", "asia_hours", "europe_hours", "america_hours"],
}
FEATURE_NAMES = [name for names in GROUPS.values() for name in names]
assert len(GROUPS) == 7 and len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == 56

WINDOWS = {"5m": (12, 48, 288), "15m": (4, 16, 96)}


def _safe_div(numerator: pd.Series, denominator: pd.Series | float) -> pd.Series:
    den = denominator.replace(0, np.nan) if isinstance(denominator, pd.Series) else denominator
    return numerator / den


def build_features(bars: pd.DataFrame, *, symbol: str, frequency: str) -> pd.DataFrame:
    """Features use only the current completed bar and prior bars of one symbol."""
    if frequency not in WINDOWS:
        raise ValueError(frequency)
    required = {"open_time", "close_time", "open", "high", "low", "close", "quote_volume", "trade_count", "taker_buy_quote_volume"}
    if not required.issubset(bars.columns):
        raise ValueError(f"missing columns: {sorted(required - set(bars.columns))}")
    if not bars.open_time.is_monotonic_increasing or bars.open_time.duplicated().any():
        raise ValueError("bars must be sorted and unique")
    fast, med, slow = WINDOWS[frequency]
    o = bars.open.astype(float)
    h = bars.high.astype(float)
    l = bars.low.astype(float)
    c = bars.close.astype(float)
    q = bars.quote_volume.astype(float)
    trades = bars.trade_count.astype(float)
    buyq = bars.taker_buy_quote_volume.astype(float)
    logc = np.log(c)
    ret = logc.diff()
    absret = ret.abs()
    ret2 = logc.diff(2)
    hour = pd.to_datetime(bars.open_time, unit="ms", utc=True)
    rng = h - l
    body = c - o
    rv_fast = ret.pow(2).rolling(fast, min_periods=fast).sum()
    rv_med = ret.pow(2).rolling(med, min_periods=med).sum()
    rv_slow = ret.pow(2).rolling(slow, min_periods=slow).sum()
    up = ret.clip(lower=0).pow(2).rolling(fast, min_periods=fast).sum()
    down = ret.clip(upper=0).pow(2).rolling(fast, min_periods=fast).sum()
    momentum_fast = logc.diff(fast)
    momentum_med = logc.diff(med)
    momentum_slow = logc.diff(slow)
    ema_fast = c.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_med = c.ewm(span=med, adjust=False, min_periods=med).mean()
    ema_slow = c.ewm(span=slow, adjust=False, min_periods=slow).mean()
    q_fast = q.rolling(fast, min_periods=fast).sum()
    q_med = q.rolling(med, min_periods=med).sum()
    buy_fast = buyq.rolling(fast, min_periods=fast).sum()
    buy_med = buyq.rolling(med, min_periods=med).sum()
    flow_fast = 2 * _safe_div(buy_fast, q_fast) - 1
    flow_med = 2 * _safe_div(buy_med, q_med) - 1
    sign = np.sign(ret)
    neg_rate = (ret < 0).astype(float).rolling(med, min_periods=med).mean()
    pos_rate = (ret > 0).astype(float).rolling(med, min_periods=med).mean()
    sign_change = (sign * sign.shift(1) < 0).astype(float)
    prev_high = h.shift(1).rolling(med, min_periods=med).max()
    prev_low = l.shift(1).rolling(med, min_periods=med).min()
    typical = (h + l + c) / 3
    rolling_vwap = _safe_div((typical * q).rolling(med, min_periods=med).sum(), q_med)
    true_range = pd.concat([rng, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    variance_1 = ret.rolling(med, min_periods=med).var()
    variance_2 = ret2.rolling(med, min_periods=med).var()
    sr = _safe_div((pos_rate - neg_rate).abs(), pos_rate + neg_rate)

    f: dict[str, pd.Series | np.ndarray] = {}
    f["rv_fast"] = rv_fast
    f["rv_med"] = rv_med
    f["rv_slow"] = rv_slow
    f["rv_fast_slow"] = _safe_div(rv_fast / fast, rv_slow / slow)
    f["rv_accel"] = _safe_div(rv_fast - rv_fast.shift(fast), rv_slow)
    f["range_vol_fast"] = np.log(h / l).pow(2).rolling(fast, min_periods=fast).sum()
    f["body_vol_fast"] = np.log(c / o).pow(2).rolling(fast, min_periods=fast).sum()
    f["rv_surprise"] = _safe_div(rv_fast, rv_fast.rolling(slow, min_periods=slow).mean())

    f["up_semivar_fast"] = up
    f["down_semivar_fast"] = down
    f["semivar_balance"] = _safe_div(up - down, up + down)
    small_variance = rv_slow * (fast / slow) * 0.01 + 1e-12
    f["down_up_ratio"] = np.log((down + small_variance) / (up + small_variance))
    f["negative_rate_med"] = neg_rate
    f["return_vol_corr_med"] = ret.rolling(med, min_periods=med).corr(ret.pow(2))
    f["upper_wick_med"] = _safe_div(h - pd.concat([o, c], axis=1).max(axis=1), rng).rolling(med, min_periods=med).mean()
    f["lower_wick_med"] = _safe_div(pd.concat([o, c], axis=1).min(axis=1) - l, rng).rolling(med, min_periods=med).mean()

    f["body_range"] = _safe_div(body, rng)
    f["close_location"] = _safe_div(c - l, rng) * 2 - 1
    f["range_ratio"] = _safe_div(rng.rolling(fast, min_periods=fast).mean(), rng.rolling(med, min_periods=med).mean())
    f["open_gap_z"] = _safe_div(np.log(o / c.shift(1)), _safe_div(rv_med, med).pow(0.5))
    f["vwap_distance"] = np.log(c / rolling_vwap)
    f["above_prev_high"] = _safe_div(c - prev_high, c) * (c > prev_high)
    f["below_prev_low"] = _safe_div(prev_low - c, c) * (c < prev_low)
    f["reversal_from_high"] = _safe_div(c - h.rolling(med, min_periods=med).max(), c)

    f["momentum_fast"] = momentum_fast
    f["momentum_med"] = momentum_med
    f["momentum_slow"] = momentum_slow
    f["momentum_z_med"] = _safe_div(momentum_med, rv_med.pow(0.5))
    f["ema_fast_med"] = np.log(ema_fast / ema_med)
    f["ema_med_slow"] = np.log(ema_med / ema_slow)
    f["direction_alignment"] = (np.sign(momentum_fast) + np.sign(momentum_med) + np.sign(momentum_slow)) / 3
    f["breakout_balance"] = _safe_div(c - prev_high, c) - _safe_div(prev_low - c, c)

    f["efficiency_fast"] = _safe_div(momentum_fast.abs(), absret.rolling(fast, min_periods=fast).sum())
    f["efficiency_med"] = _safe_div(momentum_med.abs(), absret.rolling(med, min_periods=med).sum())
    f["sign_flip_med"] = sign_change.rolling(med, min_periods=med).mean()
    f["return_acf1_med"] = ret.rolling(med, min_periods=med).corr(ret.shift(1))
    f["variance_ratio_2"] = _safe_div(variance_2, 2 * variance_1)
    f["choppiness_med"] = _safe_div(true_range.rolling(med, min_periods=med).sum(), h.rolling(med, min_periods=med).max() - l.rolling(med, min_periods=med).min())
    p = pos_rate.clip(1e-8, 1 - 1e-8)
    f["sign_entropy_med"] = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    f["path_range_ratio"] = _safe_div(momentum_med.abs(), np.log(h.rolling(med, min_periods=med).max() / l.rolling(med, min_periods=med).min())) * sr

    f["quote_rel_fast"] = _safe_div(q_fast / fast, q.rolling(slow, min_periods=slow).mean())
    f["quote_rel_slow"] = _safe_div(q_med / med, q.rolling(slow, min_periods=slow).mean())
    f["trades_rel_fast"] = _safe_div(trades.rolling(fast, min_periods=fast).mean(), trades.rolling(slow, min_periods=slow).mean())
    f["mean_trade_quote"] = _safe_div(q_fast, trades.rolling(fast, min_periods=fast).sum())
    f["flow_imb_fast"] = flow_fast
    f["flow_imb_med"] = flow_med
    f["flow_change"] = flow_fast - flow_med
    f["impact_proxy"] = _safe_div(absret.rolling(fast, min_periods=fast).sum(), q_fast)

    hour_decimal = hour.dt.hour + hour.dt.minute / 60
    week_decimal = hour.dt.dayofweek + hour_decimal / 24
    f["hour_sin"] = np.sin(2 * np.pi * hour_decimal / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour_decimal / 24)
    f["week_sin"] = np.sin(2 * np.pi * week_decimal / 7)
    f["week_cos"] = np.cos(2 * np.pi * week_decimal / 7)
    f["is_weekend"] = (hour.dt.dayofweek >= 5).astype(float)
    f["asia_hours"] = hour.dt.hour.between(0, 7).astype(float)
    f["europe_hours"] = hour.dt.hour.between(8, 15).astype(float)
    f["america_hours"] = hour.dt.hour.between(16, 23).astype(float)

    if set(f) != set(FEATURE_NAMES):
        raise AssertionError("feature registry and implementation diverged")
    matrix = pd.DataFrame({name: np.asarray(f[name], dtype=np.float32) for name in FEATURE_NAMES})
    matrix.replace([np.inf, -np.inf], np.nan, inplace=True)
    matrix.insert(0, "frequency", frequency)
    matrix.insert(0, "symbol", symbol)
    matrix.insert(0, "available_at", pd.to_datetime(bars.close_time.astype("int64") + 1, unit="ms", utc=True))
    return matrix
