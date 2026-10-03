"""Causal RV/BV/HAR-J and intraday volatility normalization."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.optimize import lsq_linear
from statsmodels.nonparametric.smoothers_lowess import lowess


TIER_A = ("ES", "NQ", "RTY")
EXPECTED_CORE_RETURNS = {"ES": 189, "NQ": 189, "RTY": 189, "HSI": 54, "HTI": 54}


def bipower_variation(returns: np.ndarray) -> float:
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2:
        return np.nan
    return float(np.pi / 2 * n / (n - 1) * np.sum(np.abs(values[1:]) * np.abs(values[:-1])))


def _subsample_returns(group: pd.DataFrame, offset: int) -> np.ndarray:
    sample = group.iloc[offset::5]
    returns = np.log(sample["close"] / sample["close"].shift())
    valid = (
        (~sample["is_filled"])
        & (~sample["is_filled"].shift(fill_value=True))
        & sample["local_symbol"].eq(sample["local_symbol"].shift())
        & sample["timestamp"].diff().eq(pd.Timedelta(minutes=5))
    )
    return returns.where(valid).dropna().to_numpy()


def compute_session_variation(minute: pd.DataFrame) -> pd.DataFrame:
    """Compute comparable core-window session RV and BV.

    Stage 1 found systematic session-tail truncation. The fixed windows below
    are the intersection observed on every normal session, preventing coverage
    length from masquerading as a volatility innovation.
    """
    work = minute.copy()
    cme = work["root"].isin(TIER_A)
    cme_elapsed = (work["local_minute"] - 18 * 60) % (24 * 60)
    cme_core = cme & cme_elapsed.le(946)
    hk_core = (~cme) & work["local_minute"].between(17 * 60 + 15, 21 * 60 + 46)
    work = work.loc[cme_core | hk_core].sort_values(["root", "session_id", "timestamp"])

    rows = []
    for (root, session_id), group in work.groupby(["root", "session_id"], sort=True, observed=True):
        offset_returns = [_subsample_returns(group, offset) for offset in range(5)]
        rv_components = [np.square(values).sum() for values in offset_returns if len(values)]
        base_returns = offset_returns[0]
        rv = float(np.mean(rv_components)) if rv_components else np.nan
        bv = bipower_variation(base_returns)
        valid_close = group.loc[~group["is_filled"], "close"]
        session_return = float(np.log(valid_close.iloc[-1] / valid_close.iloc[0])) if len(valid_close) > 1 else np.nan
        minimum_returns = 180 if root in TIER_A else 50
        enough_coverage = len(base_returns) >= minimum_returns
        continuous = min(rv, bv) if enough_coverage and np.isfinite(rv) and np.isfinite(bv) else np.nan
        rows.append(
            {
                "root": root,
                "session_id": session_id,
                "rv": rv,
                "bv": bv,
                "continuous_var": continuous,
                "jump_var": max(rv - bv, 0.0) if enough_coverage and np.isfinite(rv) and np.isfinite(bv) else np.nan,
                "session_ret": session_return,
                "n_core_returns": len(base_returns),
            }
        )
    return pd.DataFrame(rows)


def _har_records(daily: pd.DataFrame) -> pd.DataFrame:
    parts = []
    tiny = np.finfo(float).tiny
    for root, group in daily.groupby("root", sort=True, observed=True):
        # Keep the complete session calendar. Dropping an unusable target day
        # before shifting would let the model know, one day early, whether that
        # target will later have sufficient observations.
        group = group.sort_values("session_id").copy()
        continuous = group["continuous_var"].clip(lower=tiny)
        group["log_c"] = np.log(continuous)
        group["log_c_w"] = np.log(continuous.rolling(5, min_periods=5).mean())
        group["log_c_m"] = np.log(continuous.rolling(22, min_periods=22).mean())
        group["log1p_j"] = np.log1p(group["jump_var"])
        group["r_neg"] = group["session_ret"].clip(upper=0)
        group["target_log_c"] = group["log_c"].shift(periods=-1)
        group["target_session"] = group["session_id"].shift(periods=-1)
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


HAR_FEATURES = ("log_c", "log_c_w", "log_c_m", "log1p_j", "r_neg")


def _design(
    records: pd.DataFrame,
    root_levels: list[str],
    means: pd.Series | None = None,
    scales: pd.Series | None = None,
) -> np.ndarray:
    columns = [np.ones(len(records))]
    features = records.loc[:, HAR_FEATURES]
    if means is not None and scales is not None:
        features = (features - means) / scales
    columns.extend(features[feature].to_numpy(dtype=float) for feature in HAR_FEATURES)
    for root in root_levels[1:]:
        columns.append(records["root"].eq(root).to_numpy(dtype=float))
    return np.column_stack(columns)


def _forecast_pool(records: pd.DataFrame, min_sessions: int) -> pd.Series:
    prediction = pd.Series(np.nan, index=records.index, dtype=float)
    roots = sorted(records["root"].unique())
    predictors = ["log_c", "log_c_w", "log_c_m", "log1p_j", "r_neg"]
    candidates = records.dropna(subset=predictors + ["target_session"])
    trainable = candidates.dropna(subset=["target_log_c"])
    for target_date in sorted(candidates["target_session"].unique()):
        train = trainable.loc[trainable["target_session"].lt(target_date)]
        if train["target_session"].nunique() < min_sessions:
            continue
        test = candidates.loc[candidates["target_session"].eq(target_date)]
        means = train.loc[:, HAR_FEATURES].mean()
        scales = train.loc[:, HAR_FEATURES].std().replace(0, 1)
        train_design = _design(train, roots, means, scales)
        lower = np.array([-np.inf, 0, 0, 0, 0, -np.inf] + [-np.inf] * (len(roots) - 1))
        upper = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, 0] + [np.inf] * (len(roots) - 1))
        coefficients = lsq_linear(
            train_design,
            train["target_log_c"].to_numpy(),
            bounds=(lower, upper),
        ).x
        prediction.loc[test.index] = _design(test, roots, means, scales) @ coefficients
    return prediction


def expanding_har_forecast(daily: pd.DataFrame, min_sessions: int = 40) -> pd.DataFrame:
    records = _har_records(daily)
    predictor_forecast = pd.Series(np.nan, index=records.index, dtype=float)
    tier_a_mask = records["root"].isin(TIER_A)
    predictor_forecast.loc[tier_a_mask] = _forecast_pool(records.loc[tier_a_mask], min_sessions)
    for root in ("HSI", "HTI"):
        mask = records["root"].eq(root)
        if mask.any():
            predictor_forecast.loc[mask] = _forecast_pool(records.loc[mask], min_sessions)

    records["forecast_for_session"] = records["target_session"]
    records["forecast_log_var"] = predictor_forecast
    forecasts = records[["root", "forecast_for_session", "forecast_log_var"]].dropna(subset=["forecast_for_session"])
    forecasts = forecasts.rename(columns={"forecast_for_session": "session_id", "forecast_log_var": "har_log_var"})

    result = daily.sort_values(["root", "session_id"]).copy()
    result = result.merge(forecasts, on=["root", "session_id"], how="left", validate="one_to_one")
    log_c = np.log(result["continuous_var"].clip(lower=np.finfo(float).tiny))
    result["fallback_log_var"] = log_c.groupby(result["root"], sort=False).transform(
        lambda values: values.shift(1).ewm(halflife=5, adjust=False, min_periods=5).mean()
    )
    result["forecast_log_var"] = result["har_log_var"].fillna(result["fallback_log_var"])
    result["har_ready"] = result["har_log_var"].notna()
    expected = result["root"].map(EXPECTED_CORE_RETURNS).astype(float)
    result["sigma_day"] = np.sqrt(np.exp(result["forecast_log_var"]) / expected)
    return result.sort_values(["root", "session_id"]).reset_index(drop=True)


def estimate_causal_seasonality(
    bars: pd.DataFrame,
    *,
    min_sessions: int = 20,
    update_every: int = 5,
    smooth_width: int = 5,
    lowess_frac: float = 0.10,
) -> pd.Series:
    output = pd.Series(1.0, index=bars.index, name="seas", dtype=float)
    event_minutes = {8 * 60 + 30, 9 * 60 + 30, 14 * 60, 15 * 60 + 50, 15 * 60 + 55, 16 * 60}
    for root, root_bars in bars.groupby("root", sort=False, observed=True):
        sessions = list(pd.unique(root_bars["session_id"]))
        current_curve: pd.Series | None = None
        for number, session_id in enumerate(sessions):
            current = root_bars.index[root_bars["session_id"].eq(session_id)]
            if number >= min_sessions and ((number - min_sessions) % update_every == 0 or current_curve is None):
                history_sessions = set(sessions[:number])
                history = root_bars.loc[root_bars["session_id"].isin(history_sessions)].copy()
                if "is_vol_core" in history:
                    history = history.loc[history["is_vol_core"]]
                if "is_half_day" in history:
                    history = history.loc[~history["is_half_day"]]
                standardized = history["ret"].abs() / history["sigma_day"]
                valid = standardized.notna() & np.isfinite(standardized)
                grouped_scale = standardized.loc[valid].groupby(history.loc[valid, "local_minute"])
                raw_curve = grouped_scale.median().sort_index()
                mean_absolute = grouped_scale.mean().reindex(raw_curve.index)
                if len(raw_curve) >= 3:
                    filtered = median_filter(raw_curve.to_numpy(), size=smooth_width, mode="nearest")
                    smoothed = lowess(filtered, raw_curve.index.to_numpy(), frac=lowess_frac, return_sorted=False)
                    current_curve = pd.Series(smoothed, index=raw_curve.index)
                    # The median defines the robust shape. A bounded first-moment
                    # calibration aligns the final scale with the G2 E|r| target.
                    mean_correction = (mean_absolute / current_curve).clip(0.4, 2.5)
                    current_curve = current_curve * mean_correction
                    if root in TIER_A:
                        preserved = raw_curve.index.intersection(event_minutes)
                        # Preserve predictable event-time peaks on the target
                        # first-moment scale instead of flattening them away.
                        current_curve.loc[preserved] = mean_absolute.loc[preserved]
                    rms = float(np.sqrt(np.mean(np.square(current_curve))))
                    current_curve = current_curve / rms
            if current_curve is not None:
                slots = root_bars.loc[current, "local_minute"].to_numpy()
                output.loc[current] = np.interp(
                    slots,
                    current_curve.index.to_numpy(dtype=float),
                    current_curve.to_numpy(dtype=float),
                    left=float(current_curve.iloc[0]),
                    right=float(current_curve.iloc[-1]),
                )
    return output


def finalize_sigma_hat(
    bars: pd.DataFrame,
    *,
    ewma_lambda: float = 0.97,
    theta: float = 0.4,
    clip: tuple[float, float] = (0.5, 2.5),
) -> pd.DataFrame:
    result = bars.copy()
    base_sigma = result["sigma_day"] * result["seas"]
    residual = result["ret"] / base_sigma
    past_variance = residual.pow(2).groupby(result["root"], sort=False).transform(
        lambda values: values.ewm(alpha=1 - ewma_lambda, adjust=False, min_periods=5).mean().shift(1)
    )
    result["u"] = past_variance.pow(theta / 2).clip(*clip).fillna(1.0)
    result["sigma_hat"] = base_sigma * result["u"]
    return result


def har_oos_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for root, group in daily.groupby("root", sort=True, observed=True):
        group = group.sort_values("session_id").copy()
        group["actual_log_var"] = np.log(group["continuous_var"])
        group["historical_mean"] = group["actual_log_var"].expanding().mean().shift(1)
        valid = group.dropna(subset=["har_log_var", "actual_log_var", "historical_mean"]).copy()
        actual = valid["actual_log_var"]
        squared_error = np.square(actual - valid["har_log_var"])
        benchmark_error = np.square(actual - valid["historical_mean"])
        r2 = 1 - squared_error.sum() / benchmark_error.sum() if len(valid) else np.nan
        rows.append({"root": root, "oos_r2": float(r2), "n_oos": len(valid)})
    return pd.DataFrame(rows)


def seasonality_flatness(vol_bars: pd.DataFrame, min_count: int = 20) -> pd.DataFrame:
    rows = []
    work = vol_bars.loc[vol_bars["seasonality_ready"]].copy()
    work["abs_z"] = (work["ret"] / work["sigma_hat"]).abs()
    for root, group in work.groupby("root", sort=True, observed=True):
        curve = group.groupby("local_minute")["abs_z"].agg(["mean", "count"])
        curve = curve.loc[curve["count"].ge(min_count)].copy()
        elapsed = (
            (curve.index.to_numpy() - 18 * 60) % (24 * 60)
            if root in TIER_A
            else (curve.index.to_numpy() - 17 * 60) % (24 * 60)
        )
        curve["elapsed"] = elapsed
        curve = curve.sort_values("elapsed")
        smoothed = lowess(curve["mean"], curve["elapsed"], frac=0.10, return_sorted=False)
        raw_range = (curve["mean"].max() - curve["mean"].min()) / curve["mean"].mean()
        smooth_range = (smoothed.max() - smoothed.min()) / smoothed.mean()
        rows.append(
            {
                "root": root,
                "raw_range_over_mean": float(raw_range),
                "smoothed_range_over_mean": float(smooth_range),
                "slots": len(curve),
            }
        )
    return pd.DataFrame(rows)
