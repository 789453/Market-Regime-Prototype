from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_stage4_signal_contracts_and_safe_fallback():
    summary = pd.read_csv(ROOT / "reports" / "tables" / "04_regime_summary.csv").set_index("detector_id")
    for number in range(1, 9):
        detector_id = f"D{number:02d}"
        path = ROOT / "data" / "processed" / f"signals_stage4_{detector_id.lower()}.parquet"
        signal = pd.read_parquet(path)
        assert len(signal) == 99_119 and signal.index.is_unique
        assert signal["confidence"].between(0, 1).all()
        assert signal["raw_gate_score"].between(0.2, 1.5).all()
        assert signal["gate_score"].between(0.2, 1.5).all()
        assert signal.loc[signal["direction"].eq(0), "confidence"].eq(0).all()
        if summary.loc[detector_id, "applied_mode"] == "unconditional":
            assert signal["gate_score"].eq(1.0).all()


def test_stage4_hierarchy_and_prefix_checks():
    effects = pd.read_parquet(ROOT / "reports" / "tables" / "04_hierarchical_effects.parquet")
    prefix = pd.read_csv(ROOT / "reports" / "tables" / "04_causal_prefix_checks.csv")
    assert effects["term_order"].between(1, 2).all()
    assert effects["term_order"].max() == 2
    assert prefix["passed"].all()
