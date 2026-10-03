"""Independent checks on saved signal-chain outputs and raw execution prices."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/crypto/signal_chain_v1"
SOURCE = Path(yaml.safe_load((ROOT / "configs/crypto_predictive_states.yaml").read_text(
    encoding="utf-8"))["source"])


def main() -> None:
    forecast = pd.concat([pd.read_parquet(OUT / f"forecast_rows_{phase}.parquet")
                          for phase in ("2025H2_development", "2026_exploratory")])
    book = pd.read_parquet(OUT / "portfolio_daily.parquet")
    positions = pd.read_parquet(OUT / "coin_positions_daily.parquet")
    assert not forecast.duplicated(["phase", "available_at", "symbol"]).any()
    assert (forecast.groupby(["phase", "available_at"]).size() == 12).all()
    for horizon in (4, 24):
        probability = forecast[[f"return{horizon}_bin_p{j}" for j in range(5)]].to_numpy()
        assert np.max(np.abs(probability.sum(axis=1) - 1)) < 1e-10
        assert ((forecast[f"return{horizon}_bin_actual"] >= 0) &
                (forecast[f"return{horizon}_bin_actual"] <= 4)).all()
    assert not book.isna().any().any() and not positions.isna().any().any()
    assert (positions.groupby(["phase", "strategy", "available_at"]).weight
            .apply(lambda x: x.abs().sum()) <= 1 + 1e-8).all()
    pair = positions[positions.strategy.str.startswith("pair_")]
    max_beta = pair.groupby(["phase", "strategy", "available_at"]).apply(
        lambda x: abs(np.sum(x.weight * x.beta_past)), include_groups=False).max()
    assert max_beta < 1e-12
    assert book.groupby(["phase", "available_at"]).buy_hold_wealth.nunique().max() == 1
    last = book.groupby(["phase", "strategy", "fee_bp_side"]).tail(1)
    fee_order = last.pivot(index=["phase", "strategy"], columns="fee_bp_side", values="net_wealth")
    assert (fee_order[0] >= fee_order[4] - 1e-12).all()
    assert (fee_order[4] >= fee_order[10] - 1e-12).all()
    raw_checked = []
    for symbol in ("BTCUSDT", "AVAXUSDT"):
        raw = pq.read_table(SOURCE / symbol / "5m.parquet", columns=["open_time", "open"]).to_pandas()
        stamp = raw.open_time.to_numpy(dtype=np.int64)
        price = np.log(raw.open.to_numpy(dtype=np.float64))
        for phase in forecast.phase.unique():
            sub = forecast[(forecast.symbol == symbol) & (forecast.phase == phase)]
            for item in sub.iloc[[0, len(sub) // 2, -1]].itertuples():
                entry = int((item.available_at + pd.Timedelta(minutes=5)).timestamp() * 1000)
                ix = np.searchsorted(stamp, entry)
                assert stamp[ix] == entry and stamp[ix + 288] == entry + 86_400_000
                observed = price[ix + 288] - price[ix]
                assert abs(observed - item.return24) < 1e-11
                raw_checked.append({"phase": phase, "symbol": symbol,
                                    "available_at": str(item.available_at),
                                    "difference": float(observed - item.return24)})
    result = {"forecast_rows": len(forecast), "portfolio_rows": len(book),
              "position_rows": len(positions), "max_pair_beta_exposure": float(max_beta),
              "raw_24h_return_spot_checks": raw_checked,
              "status": "pass"}
    (OUT / "integrity_checks.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "raw_24h_return_spot_checks"}))


if __name__ == "__main__":
    main()
