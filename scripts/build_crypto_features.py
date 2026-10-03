"""Build 5m/15m causal feature snapshots at every 15m bar close."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.crypto.features import FEATURE_NAMES, GROUPS, build_features


CONFIG = yaml.safe_load((ROOT / "configs/crypto_research.yaml").read_text(encoding="utf-8"))
SOURCE = Path(CONFIG["data"]["root"])
OUTPUT = ROOT / "data/crypto/features"
RAW_COLUMNS = ["open_time", "close_time", "open", "high", "low", "close", "quote_volume", "trade_count", "taker_buy_quote_volume"]


def build_symbol(symbol: str) -> dict:
    frames = {}
    for frequency in ("5m", "15m"):
        raw = pq.read_table(SOURCE / symbol / f"{frequency}.parquet", columns=RAW_COLUMNS).to_pandas()
        frames[frequency] = build_features(raw, symbol=symbol, frequency=frequency)
    fast = frames["5m"].iloc[2::3].drop(columns=["symbol", "frequency"])
    slow = frames["15m"].drop(columns=["frequency"])
    fast = fast.rename(columns={name: f"{name}_5m" for name in FEATURE_NAMES})
    slow = slow.rename(columns={name: f"{name}_15m" for name in FEATURE_NAMES})
    joined = slow.merge(fast, on="available_at", how="left", validate="one_to_one")
    if len(joined) != len(slow) or joined["hour_sin_5m"].isna().any():
        raise AssertionError(f"5m/15m close times do not align for {symbol}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / f"{symbol}.parquet"
    joined.to_parquet(target, index=False, compression="zstd")
    return {
        "symbol": symbol, "rows": len(joined), "columns": len(joined.columns),
        "first_available_at": str(joined.available_at.iloc[0]),
        "last_available_at": str(joined.available_at.iloc[-1]),
        "path": str(target),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="*", default=CONFIG["data"]["symbols"])
    args = parser.parse_args()
    unknown = set(args.symbols) - set(CONFIG["data"]["symbols"])
    if unknown:
        parser.error(f"unknown symbols: {sorted(unknown)}")
    results = [build_symbol(symbol) for symbol in args.symbols]
    manifest_path = OUTPUT / "manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    by_symbol = {item["symbol"]: item for item in previous.get("snapshots", [])}
    by_symbol.update({item["symbol"]: item for item in results})
    metadata = {"groups": GROUPS, "features_per_frequency": len(FEATURE_NAMES), "snapshots": [by_symbol[s] for s in sorted(by_symbol)],
                "available_at": "bar close_time + 1ms; both frequencies fully closed",
                "decision_clock": "every 15m close; feature table contains no future label"}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
