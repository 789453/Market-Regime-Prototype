from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_all_stage3_signal_contracts():
    for number in range(1, 9):
        detector_id = f"d{number:02d}"
        signal = pd.read_parquet(ROOT / "data" / "processed" / f"signals_{detector_id}.parquet")
        assert len(signal) == 99_119
        assert signal.index.names == ["root", "timestamp"]
        assert signal.index.is_unique
        assert signal["direction"].isin([-1, 0, 1]).all()
        assert signal["magnitude"].between(0, 1).all()
        assert signal["confidence"].eq(1.0).all()
        assert (ROOT / "data" / "processed" / f"signals_{detector_id}_manifest.json").is_file()


def test_d01_d02_artifacts_are_mutually_exclusive():
    first = pd.read_parquet(ROOT / "data" / "processed" / "signals_d01.parquet")
    second = pd.read_parquet(ROOT / "data" / "processed" / "signals_d02.parquet")
    assert not (first["direction"].ne(0) & second["direction"].ne(0)).any()
