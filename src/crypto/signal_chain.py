"""Causal daily decision rules and a marked-to-market multi-asset ledger."""

from __future__ import annotations

import numpy as np


def hysteresis(value: float, old: int, enter: float, leave: float) -> int:
    """Keep an existing sign until evidence weakens; flip only past entry."""
    if value >= enter:
        return 1
    if value <= -enter:
        return -1
    if old and old * value > leave:
        return old
    return 0


def beta_neutral_pair(long: int, short: int, beta: np.ndarray) -> np.ndarray:
    """Gross-one pair with zero exposure to the estimated common factor."""
    out = np.zeros(len(beta), dtype=np.float64)
    a, b = float(beta[long]), float(beta[short])
    out[long], out[short] = b / (a + b), -a / (a + b)
    return out


def market_weights(score: np.ndarray, enter: float, leave: float,
                   *, risk: np.ndarray | None = None, risk_limit: float = 0.6) -> np.ndarray:
    n, coins = len(score), 12
    weights = np.zeros((n, coins), dtype=np.float64)
    old = 0
    for t, value in enumerate(score):
        old = hysteresis(float(value), old, enter, leave)
        if risk is not None and risk[t] > risk_limit:
            old = 0
        weights[t] = old / coins
    return weights


def pair_weights(score: np.ndarray, beta: np.ndarray, enter: float,
                 leave: float) -> np.ndarray:
    n, coins = score.shape
    weights = np.zeros((n, coins), dtype=np.float64)
    old_long = old_short = -1
    for t in range(n):
        long = int(np.argmax(score[t]))
        short = int(np.argmin(score[t]))
        gap = float(score[t, long] - score[t, short])
        if gap >= enter:
            old_long, old_short = long, short
        elif gap < leave:
            old_long = old_short = -1
        if old_long >= 0:
            weights[t] = beta_neutral_pair(old_long, old_short, beta[t])
    return weights


def ledger(weights: np.ndarray, log_price: np.ndarray,
           fee_bp_side: float = 0.0) -> dict[str, np.ndarray]:
    """Weights set at next 5m open; each row earns until following open.

    log_price has one extra terminal price row. Constant weights are target
    fractions of current equity; per-interval arithmetic returns compound.
    Fees are turnover in target fractions, charged at the decision open.
    """
    if log_price.shape != (len(weights) + 1, weights.shape[1]):
        raise ValueError("price rows must be one longer than weight rows")
    if not np.isfinite(weights).all() or not np.isfinite(log_price).all():
        raise ValueError("nonfinite prices or weights")
    if np.max(np.abs(weights).sum(axis=1)) > 1 + 1e-8:
        raise ValueError("gross exposure exceeds one")
    simple = np.expm1(np.diff(log_price, axis=0))
    gross = (weights * simple).sum(axis=1)
    turnover = np.empty(len(weights), dtype=np.float64)
    drifted = np.zeros(weights.shape[1], dtype=np.float64)
    for t in range(len(weights)):
        turnover[t] = np.abs(weights[t] - drifted).sum()
        drifted = weights[t] * (1 + simple[t]) / (1 + gross[t])
    net = (1 - turnover * fee_bp_side / 1e4) * (1 + gross) - 1
    gross_wealth = np.cumprod(1 + gross)
    net_wealth = np.cumprod(1 + net)
    return {"gross_return": gross, "net_return": net, "turnover": turnover,
            "gross_wealth": gross_wealth, "net_wealth": net_wealth,
            "asset_contribution": weights * simple}


def buy_hold(log_price: np.ndarray) -> np.ndarray:
    """Equal starting dollars, no rebalance, including terminal mark."""
    return np.mean(np.exp(log_price - log_price[0]), axis=1)
