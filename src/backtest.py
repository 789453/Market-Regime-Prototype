"""Explicit causal event-loop backtester for Stage 5."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _benchmark(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name == "mid":
        return ((frame["open"] + frame["close"]) / 2.0).to_numpy(dtype=float)
    if name not in {"open", "wap"}:
        raise ValueError(f"unsupported execution benchmark: {name}")
    return frame[name].to_numpy(dtype=float)


def run_event_backtest(
    bars: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    execution_price: str = "wap",
    cost_multiplier: float = 1.5,
    delay_bars: int = 1,
    avoid_session_edge_bars: int = 2,
    avoid_roll_boundary_bars: int = 3,
    max_volume_participation: float = 0.05,
) -> pd.DataFrame:
    """Run close-decision/next-valid-bar execution with explicit pending orders.

    At an execution bar, the old position earns the return up to the benchmark
    and the new position earns the benchmark-to-close return. This prevents a
    close-observed decision from receiving the already elapsed part of that bar.
    """
    if delay_bars < 0:
        raise ValueError("delay_bars must be non-negative")
    idx = bars.index.intersection(decisions.index)
    b = bars.reindex(idx)
    d = decisions.reindex(idx)
    pieces: list[pd.DataFrame] = []
    for _, frame in b.groupby(level="root", sort=False):
        dec = d.reindex(frame.index)
        n = len(frame)
        benchmark = _benchmark(frame, execution_price)
        close = frame["close"].to_numpy(dtype=float)
        ret = frame["ret"].to_numpy(dtype=float)
        overnight = frame["overnight_ret"].to_numpy(dtype=float)
        total_return = np.where(np.isfinite(ret), ret, np.where(np.isfinite(overnight), overnight, 0.0))
        valid = frame["is_valid"].fillna(False).to_numpy(dtype=bool)
        bar_in = frame["bar_in_session"].to_numpy(dtype=int)
        session = frame["session_id"].astype(str)
        session_max = frame.groupby("session_id", sort=False)["bar_in_session"].transform("max").to_numpy(dtype=int)
        phase = frame["session_phase"].astype(str).to_numpy()
        roll = frame["is_roll_boundary"].fillna(False).to_numpy(dtype=bool)
        roll_near = roll.copy()
        for offset in range(1, avoid_roll_boundary_bars + 1):
            roll_near[offset:] |= roll[:-offset]
            roll_near[:-offset] |= roll[offset:]
        volume = frame["volume"].fillna(0.0).to_numpy(dtype=float)
        target = dec["target"].fillna(0.0).to_numpy(dtype=float)
        band = dec["band"].fillna(0.0).to_numpy(dtype=float)
        signal_active = dec.get("signal_active", pd.Series(True, index=dec.index)).fillna(False).to_numpy(dtype=bool)
        rt_cost = dec["round_trip_cost_bp"].fillna(0.0).to_numpy(dtype=float) * 1e-4

        position = np.zeros(n)
        gross = np.zeros(n)
        costs = np.zeros(n)
        turnover = np.zeros(n)
        executed = np.zeros(n, dtype=bool)
        volume_limited = np.zeros(n, dtype=bool)
        requested_at = np.full(n, -1, dtype=int)
        current = 0.0
        pending_target = 0.0
        pending_band = 0.0
        pending_requested = -1
        pending_active = False

        for i in range(n):
            old = current
            source = i - delay_bars
            if source >= 0:
                # The newest decision whose latency has elapsed supersedes an
                # older blocked order; it may execute now or at the next valid bar.
                pending_target = target[source]
                pending_band = band[source]
                pending_requested = source
                pending_active = signal_active[source]
            can_trade = valid[i] and not roll_near[i]
            at_edge = avoid_session_edge_bars > 0 and (
                bar_in[i] < avoid_session_edge_bars
                or bar_in[i] > session_max[i] - avoid_session_edge_bars
            )
            if pending_requested >= 0 and can_trade:
                desired = pending_target
                delta = desired - old
                opens_risk = abs(desired) > abs(old) + 1e-12 or (
                    old != 0 and desired != 0 and np.sign(desired) != np.sign(old)
                )
                if opens_risk and (at_edge or phase[i] == "HK_LATE"):
                    delta = 0.0
                # A no-trade band may smooth an active bet, but must not trap an
                # expired detector position outside its registered horizon.
                if pending_active and abs(delta) <= pending_band:
                    delta = 0.0
                max_delta = max_volume_participation * volume[i]
                if abs(delta) > max_delta:
                    delta = np.sign(delta) * max_delta
                    volume_limited[i] = True
                if delta != 0.0:
                    current = old + delta
                    turnover[i] += abs(delta)
                    costs[i] += 0.5 * rt_cost[i] * cost_multiplier * abs(delta)
                    executed[i] = True
                    requested_at[i] = pending_requested
                pending_requested = -1

            if executed[i] and benchmark[i] > 0 and close[i] > 0:
                post_return = np.log(close[i] / benchmark[i])
                pre_return = total_return[i] - post_return
                gross[i] = old * pre_return + current * post_return
            else:
                gross[i] = old * total_return[i]

            if roll[i] and old != 0.0:
                # Close and reopen the carried position: one full round trip.
                costs[i] += rt_cost[i] * cost_multiplier * abs(old)
                turnover[i] += 2.0 * abs(old)
            position[i] = current

        piece = pd.DataFrame(index=frame.index)
        piece["position"] = position
        piece["gross_return"] = gross
        piece["cost"] = costs
        piece["net_return"] = gross - costs
        piece["turnover"] = turnover
        piece["executed"] = executed
        piece["volume_limited"] = volume_limited
        piece["execution_delay_bars"] = np.where(executed, np.arange(n) - requested_at, np.nan)
        piece["target"] = target
        piece["session_id"] = session.to_numpy()
        piece["session_phase"] = phase
        pieces.append(piece)
    return pd.concat(pieces).sort_index()


def aggregate_portfolio(backtest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate instrument rows first by timestamp, then by local session id."""
    intraday = backtest.groupby(level="timestamp").agg(
        gross_return=("gross_return", "sum"),
        cost=("cost", "sum"),
        net_return=("net_return", "sum"),
        turnover=("turnover", "sum"),
    )
    daily_parts = []
    for (root, session_id), frame in backtest.groupby(
        [backtest.index.get_level_values("root"), "session_id"], sort=False
    ):
        daily_parts.append(
            {
                "root": root,
                "session_id": session_id,
                "gross_return": frame["gross_return"].sum(),
                "cost": frame["cost"].sum(),
                "net_return": frame["net_return"].sum(),
                "turnover": frame["turnover"].sum(),
            }
        )
    by_root_session = pd.DataFrame(daily_parts)
    by_root_session["date"] = pd.to_datetime(by_root_session["session_id"].str[:10], errors="coerce")
    daily = by_root_session.groupby("date").agg(
        gross_return=("gross_return", "sum"),
        cost=("cost", "sum"),
        net_return=("net_return", "sum"),
        turnover=("turnover", "sum"),
    )
    return intraday, daily
