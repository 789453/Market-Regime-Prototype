"""Read-only inventory of the proposed crypto OHLCV source."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


SOURCE = Path(r"D:\Trading\practical_crypto_strategy\data\parquet")
FREQUENCIES = {"5m": 300_000, "15m": 900_000, "1h": 3_600_000}


def inspect(path: Path) -> dict:
    parquet = pq.ParquetFile(path)
    columns = parquet.schema_arrow.names
    values = parquet.read(columns=["open_time", "close_time", "date", "open", "high", "low", "close", "volume", "quote_volume", "trade_count", "taker_buy_volume", "taker_buy_quote_volume", "taker_sell_volume", "taker_buy_ratio", "log_return", "range_pct", "body_pct", "vwap"]).to_pandas()
    times = values.open_time.to_numpy(dtype=np.int64)
    gaps = np.diff(times)
    expected = FREQUENCIES[path.stem]
    missing = int(np.maximum(gaps // expected - 1, 0).sum())
    return {
        "symbol": path.parent.name,
        "frequency": path.stem,
        "rows": len(values),
        "first_utc": str(values.date.iloc[0]),
        "last_utc": str(values.date.iloc[-1]),
        "column_names": columns,
        "schema": str(parquet.schema_arrow).split("-- schema metadata --")[0].strip(),
        "null_counts": {k: int(v) for k, v in values.isna().sum().items() if v},
        "duplicate_open_times": int(values.open_time.duplicated().sum()),
        "nonmonotonic_steps": int((gaps <= 0).sum()),
        "gap_events": int((gaps > expected).sum()),
        "missing_expected_bars": missing,
        "short_steps": int(((gaps > 0) & (gaps < expected)).sum()),
        "timestamp_mismatch": int((values.date.astype("int64") // 1_000_000 != values.open_time).sum()),
        "invalid_ohlc": int(((values.high < values[["open", "close", "low"]].max(axis=1)) | (values.low > values[["open", "close", "high"]].min(axis=1))).sum()),
        "nonpositive_price": int((values[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()),
        "negative_volume_or_trades": int(((values[["volume", "quote_volume", "trade_count"]] < 0).any(axis=1)).sum()),
        "zero_volume": int((values.volume == 0).sum()),
        "close_time_mismatch": int((values.close_time.to_numpy(dtype=np.int64) - times != expected - 1).sum()),
        "max_gap_bars": int(gaps.max() // expected) if len(gaps) else 0,
    }


def main() -> None:
    results = [inspect(path) for path in sorted(SOURCE.glob("*/*.parquet"))]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
