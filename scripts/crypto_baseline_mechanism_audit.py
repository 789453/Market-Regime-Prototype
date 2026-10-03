"""Read-only diagnosis of the post-hoc own-X2 80/40 historical candidate.

Uses the original saved forecasts, positions and ledgers.  All proposed
counterfactuals use the same recorded dates; this is a mechanism audit, not
an independent performance test or a reconstructed execution backtest.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/crypto/signal_chain_v1"
OUT = ROOT / "reports/crypto/baseline_mechanism_audit"
BASE = "pair_full_x2_own_sparse_posthoc"
MOM = "pair_momentum_own_sparse_same_entry"
PHASES = ("2025H2_development", "2026_exploratory")


def read_phase(phase: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    forecast = pd.read_parquet(SOURCE / f"forecast_rows_{phase}.parquet")
    book = pd.read_parquet(SOURCE / "portfolio_daily.parquet")
    book = book[(book.phase == phase) & (book.fee_bp_side == 4)]
    position = pd.read_parquet(SOURCE / "coin_positions_daily.parquet")
    position = position[(position.phase == phase) & (position.strategy == BASE)]
    return forecast, book, position


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    identity = [{"name": "完整本币 X2，80/40（后验）", "strategy_id": BASE,
                 "model": "full_x2_own_alpha_24h.joblib", "inputs": "100 X2 + 12 symbol dummies",
                 "decision": "00:00 UTC completed snapshot", "execution": "next 5m 00:05 open",
                 "threshold_source": "in-sample model prediction on preceding fit range",
                 "fee_scenario_bp_per_side": 4,
                 "saved_forecasts": "signal_chain_v1/forecast_rows_{phase}.parquet",
                 "saved_ledger": "signal_chain_v1/portfolio_daily.parquet"}]
    pd.DataFrame(identity).to_csv(OUT / "baseline_identity.csv", index=False, encoding="utf-8-sig")
    concentration, selection, scales, alignment, survival, events = [], [], [], [], [], []
    for phase in PHASES:
        forecast, book, position = read_phase(phase)
        baseline = book[book.strategy == BASE].sort_values("available_at").copy()
        same = book[book.strategy == MOM].sort_values("available_at").copy()
        reverse = book[book.strategy == "pair_reverse_own_sparse"].sort_values("available_at").copy()
        if not (len(baseline) == len(same) == len(reverse)):
            raise AssertionError("daily counterfactual clock mismatch")
        if not np.array_equal(baseline.available_at.to_numpy(), same.available_at.to_numpy()):
            raise AssertionError("counterfactual timestamps mismatch")
        active = baseline.gross_exposure.to_numpy() > .05
        for name, candidate in ((BASE, baseline), (MOM, same), ("reverse_own", reverse)):
            net = candidate.net_return.to_numpy()
            selection.append({"phase": phase, "policy": name, "days": len(net),
                              "active_days": int(active.sum()),
                              "same_entry_net_return": float(np.prod(1 + net) - 1),
                              "same_entry_net_bp_active_day": float(net[active].mean() * 1e4)})
        net = baseline.net_return.to_numpy()
        positive = np.sort(net)[::-1]
        for n in (1, 5, 10):
            concentration.append({"phase": phase, "best_days": n,
                                  "best_sum_bp": float(positive[:n].sum() * 1e4),
                                  "all_log_return": float(np.log1p(net).sum()),
                                  "without_best_log_return": float(np.log1p(net).sum() -
                                                                   np.log1p(positive[:n]).sum())})
        threshold = json.loads((SOURCE / f"thresholds_{phase}.json").read_text())
        enter = threshold["pair_full_x2_own_sparse_enter"]
        leave = threshold["pair_full_x2_own_sparse_exit"]
        weights = position.pivot(index="available_at", columns="symbol", values="weight")
        p = forecast.pivot(index="available_at", columns="symbol", values="alpha_pred24_full_x2_own")
        b = forecast.pivot(index="available_at", columns="symbol", values="beta_past")
        y = forecast.pivot(index="available_at", columns="symbol", values="return24")
        if not (weights.index.equals(p.index) and p.index.equals(b.index) and p.index.equals(y.index)):
            raise AssertionError("saved forecasts, weights and outcomes misaligned")
        pv, bv, wv, yv = p.to_numpy(), b.to_numpy(), weights.to_numpy(), y.to_numpy()
        raw_gap = np.ptp(pv, axis=1)
        long = np.argmax(wv, axis=1)
        short = np.argmin(wv, axis=1)
        candidate_long = np.argmax(pv, axis=1)
        candidate_short = np.argmin(pv, axis=1)
        held = np.abs(wv).sum(axis=1) > .05
        actual_gross = np.sum(wv * np.expm1(yv), axis=1)
        chosen_score = ((bv[np.arange(len(pv)), short] * pv[np.arange(len(pv)), long] -
                         bv[np.arange(len(pv)), long] * pv[np.arange(len(pv)), short]) /
                        (bv[np.arange(len(pv)), long] + bv[np.arange(len(pv)), short]))
        proposed_score = ((bv[np.arange(len(pv)), candidate_short] * pv[np.arange(len(pv)), candidate_long] -
                           bv[np.arange(len(pv)), candidate_long] * pv[np.arange(len(pv)), candidate_short]) /
                          (bv[np.arange(len(pv)), candidate_long] + bv[np.arange(len(pv)), candidate_short]))
        for state in sorted(forecast.past_market_regime.unique()):
            state_by_day = forecast.groupby("available_at").past_market_regime.first().reindex(p.index)
            mask = (state_by_day.to_numpy() == state)
            scales.append({"phase": phase, "regime": state, "days": int(mask.sum()),
                           "train_in_sample_enter": enter, "train_in_sample_exit": leave,
                           "forward_score_q50": float(np.quantile(raw_gap[mask], .5)),
                           "forward_score_q80": float(np.quantile(raw_gap[mask], .8)),
                           "forward_above_train_enter": float((raw_gap[mask] >= enter).mean()),
                           "actual_active": float(held[mask].mean())})
        for q in range(5):
            rank = pd.qcut(pd.Series(raw_gap).rank(method="first"), 5, labels=False).to_numpy() == q
            selected = rank & held
            alignment.append({"phase": phase, "gap_quintile": q + 1, "days": int(rank.sum()),
                              "held_days": int(selected.sum()),
                              "mean_raw_gap_bp": float(raw_gap[rank].mean() * 1e4),
                              "mean_beta_weighted_candidate_bp": float(proposed_score[rank].mean() * 1e4),
                              "actual_gross_bp_held": float(actual_gross[selected].mean() * 1e4)
                              if selected.any() else np.nan})
        prior_pair = np.r_[False, (long[1:] == long[:-1]) & (short[1:] == short[:-1])]
        for t in range(len(pv)):
            if not held[t]:
                continue
            events.append({"phase": phase, "available_at": str(p.index[t]),
                           "long": p.columns[long[t]], "short": p.columns[short[t]],
                           "raw_gap_bp": raw_gap[t] * 1e4,
                           "held_pair_pred_bp": chosen_score[t] * 1e4,
                           "candidate_pair_pred_bp": proposed_score[t] * 1e4,
                           "actual_pair_gross_bp": actual_gross[t] * 1e4,
                           "continued_from_prior": bool(prior_pair[t] and held[t - 1])})
        for continuing in (False, True):
            mask = held & prior_pair if continuing else held & ~prior_pair
            if mask.any():
                survival.append({"phase": phase, "continued": continuing, "days": int(mask.sum()),
                                 "held_pred_bp": float(chosen_score[mask].mean() * 1e4),
                                 "candidate_pred_bp": float(proposed_score[mask].mean() * 1e4),
                                 "gross_realized_bp": float(actual_gross[mask].mean() * 1e4)})
    for name, rows in (("baseline_attribution", concentration),
                       ("selection_timing_ablation", selection),
                       ("score_scale_audit", scales),
                       ("pair_score_alignment", alignment),
                       ("holding_survival_audit", survival)):
        pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False)
    pd.DataFrame(events).to_parquet(OUT / "episode_cards.parquet", index=False)
    print("baseline mechanism audit saved", OUT)


if __name__ == "__main__":
    main()
