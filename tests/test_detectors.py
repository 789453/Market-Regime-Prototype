from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from src.detectors import DETECTORS, D01, D02


ROOT = Path(__file__).resolve().parents[1]


def _sample():
    bars = pd.read_parquet(ROOT / "data" / "interim" / "bars_5m.parquet").reset_index()
    ctx = pd.read_parquet(ROOT / "data" / "processed" / "context.parquet").reset_index()
    bars = bars.groupby("root", sort=False).head(500).reset_index(drop=True)
    ctx = ctx.merge(bars[["root", "timestamp"]], on=["root", "timestamp"], how="inner")
    return bars, ctx


def test_all_detectors_are_prefix_invariant():
    bars, ctx = _sample()
    for detector_type in DETECTORS:
        detector = detector_type()
        full = detector.detect(bars, ctx)
        for fraction in (0.4, 0.6, 0.8):
            timestamps = bars["timestamp"].drop_duplicates().sort_values()
            cutoff = timestamps.iloc[int(len(timestamps) * fraction)]
            partial_bars = bars.loc[bars["timestamp"].le(cutoff)].reset_index(drop=True)
            partial_ctx = ctx.merge(partial_bars[["root", "timestamp"]], on=["root", "timestamp"], how="inner")
            partial = detector.detect(partial_bars, partial_ctx)
            expected = full.merge(partial_bars[["root", "timestamp"]], on=["root", "timestamp"], how="inner")
            assert_frame_equal(partial, expected, check_dtype=False)


def test_breakout_branches_are_mutually_exclusive():
    bars, ctx = _sample()
    sweep = D01().detect(bars, ctx)["direction"].ne(0)
    breakout = D02().detect(bars, ctx)["direction"].ne(0)
    assert not (sweep & breakout).any()
