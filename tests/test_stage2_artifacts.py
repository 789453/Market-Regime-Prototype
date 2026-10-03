from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_stage2_artifact_contracts():
    context = pd.read_parquet(ROOT / "data" / "processed" / "context.parquet")
    labels = pd.read_parquet(ROOT / "data" / "processed" / "labels.parquet")
    assert context.index.names == ["root", "timestamp"]
    assert labels.index.names == ["root", "timestamp"]
    assert len(context) == len(labels) == 99_119
    assert str(context.index.get_level_values("timestamp").dtype) == "datetime64[ns, UTC]"
    assert context.index.is_unique and labels.index.is_unique
    assert context["gate_score"].dropna().between(0.2, 1.5).all()
    for horizon in (6, 12, 24, 48, 96):
        assert {f"fwd_ret_{horizon}", f"fwd_ret_z_{horizon}", f"label_valid_{horizon}"} <= set(labels)
        assert labels[f"label_valid_{horizon}"].dtype == bool


def test_stage2_manifests_exist():
    assert (ROOT / "data" / "processed" / "context_manifest.json").is_file()
    assert (ROOT / "data" / "processed" / "labels_manifest.json").is_file()

