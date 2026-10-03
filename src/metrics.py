"""Performance, uncertainty, multiplicity and bootstrap metrics for Stage 5."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew


def drawdown_statistics(returns: pd.Series) -> dict[str, float]:
    wealth = np.exp(returns.fillna(0.0).cumsum())
    drawdown = wealth / wealth.cummax() - 1.0
    max_dd = float(drawdown.min()) if len(drawdown) else np.nan
    underwater = drawdown < 0
    max_duration = 0
    current = 0
    recovery = np.nan
    trough_pos = int(np.argmin(drawdown.to_numpy())) if len(drawdown) else 0
    for value in underwater:
        current = current + 1 if value else 0
        max_duration = max(max_duration, current)
    if len(drawdown):
        future = np.flatnonzero(drawdown.to_numpy()[trough_pos + 1 :] >= 0)
        if len(future):
            recovery = float(future[0] + 1)
    return {
        "max_drawdown": max_dd,
        "max_drawdown_duration_days": float(max_duration),
        "recovery_days_from_max_drawdown": recovery,
    }


def sharpe_uncertainty(returns: pd.Series, annualization: int = 252) -> dict[str, float]:
    x = returns.dropna().to_numpy(dtype=float)
    if len(x) < 3 or np.std(x, ddof=1) == 0:
        return {"sharpe": np.nan, "sharpe_se": np.nan, "sharpe_ci_low": np.nan, "sharpe_ci_high": np.nan}
    sr_daily = float(np.mean(x) / np.std(x, ddof=1))
    sr = sr_daily * np.sqrt(annualization)
    sk = float(skew(x, bias=False))
    ku = float(kurtosis(x, fisher=False, bias=False))
    variance = max(1.0 - sk * sr_daily + (ku - 1.0) * sr_daily**2 / 4.0, 0.0)
    se = np.sqrt(variance / (len(x) - 1)) * np.sqrt(annualization)
    return {
        "sharpe": sr,
        "sharpe_se": float(se),
        "sharpe_ci_low": float(sr - 1.96 * se),
        "sharpe_ci_high": float(sr + 1.96 * se),
    }


def performance_summary(returns: pd.Series, annualization: int = 252) -> dict[str, float]:
    x = returns.dropna().astype(float)
    if x.empty:
        return {"annualized_return": np.nan, "annualized_volatility": np.nan}
    years = len(x) / annualization
    cumulative = float(np.exp(x.sum()) - 1.0)
    annual_return = float(np.exp(x.sum() / years) - 1.0) if years > 0 else np.nan
    vol = float(x.std(ddof=1) * np.sqrt(annualization))
    downside = x.clip(upper=0).pow(2).mean() ** 0.5 * np.sqrt(annualization)
    positive = x[x > 0]
    negative = x[x < 0]
    dd = drawdown_statistics(x)
    sharpe = sharpe_uncertainty(x, annualization)
    result = {
        "annualized_return": annual_return,
        "cumulative_return": cumulative,
        "annualized_volatility": vol,
        "annualized_downside_volatility": float(downside),
        "var_95": float(-x.quantile(0.05)),
        "var_99": float(-x.quantile(0.01)),
        "skewness": float(x.skew()),
        "kurtosis": float(x.kurtosis()),
        "sortino": float(x.mean() * annualization / downside) if downside > 0 else np.nan,
        "calmar": float(annual_return / abs(dd["max_drawdown"])) if dd["max_drawdown"] < 0 else np.nan,
        "hit_rate": float((x > 0).mean()),
        "payoff_ratio": float(positive.mean() / abs(negative.mean())) if len(positive) and len(negative) else np.nan,
    }
    result.update(dd)
    result.update(sharpe)
    return result


def stationary_block_bootstrap(
    returns: pd.Series,
    draws: int = 1000,
    expected_block: int = 5,
    seed: int = 20260905,
) -> pd.DataFrame:
    """Stationary bootstrap of annual return and Sharpe."""
    x = returns.dropna().to_numpy(dtype=float)
    n = len(x)
    rng = np.random.default_rng(seed)
    records = np.empty((draws, 2), dtype=float)
    if n == 0:
        return pd.DataFrame(records * np.nan, columns=["annualized_return", "sharpe"])
    restart = 1.0 / expected_block
    for draw in range(draws):
        sample = np.empty(n)
        index = int(rng.integers(n))
        for i in range(n):
            if i == 0 or rng.random() < restart:
                index = int(rng.integers(n))
            else:
                index = (index + 1) % n
            sample[i] = x[index]
        sd = sample.std(ddof=1)
        records[draw, 0] = np.expm1(sample.mean() * 252)
        records[draw, 1] = sample.mean() / sd * np.sqrt(252) if sd > 0 else np.nan
    return pd.DataFrame(records, columns=["annualized_return", "sharpe"])


def deflated_sharpe_ratio(returns: pd.Series, trials: int) -> dict[str, float]:
    """Bailey-Lopez de Prado style expected-maximum Sharpe deflation."""
    stats = sharpe_uncertainty(returns)
    sr = stats["sharpe"]
    se = stats["sharpe_se"]
    if not np.isfinite(sr) or not np.isfinite(se) or se <= 0:
        return {"sr0": np.nan, "deflated_sharpe_probability": np.nan}
    k = max(int(trials), 1)
    euler_gamma = 0.5772156649015329
    if k == 1:
        sr0 = 0.0
    else:
        sr0 = se * (
            (1 - euler_gamma) * norm.ppf(1 - 1 / k)
            + euler_gamma * norm.ppf(1 - 1 / (k * np.e))
        )
    return {"sr0": float(sr0), "deflated_sharpe_probability": float(norm.cdf((sr - sr0) / se))}
