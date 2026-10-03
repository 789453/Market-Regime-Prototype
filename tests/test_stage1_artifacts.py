from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_bars_5m_artifact_meets_g1_contract() -> None:
    bars = pd.read_parquet(ROOT / "data" / "interim" / "bars_5m.parquet")
    required = {
        "local_symbol", "contract_con_id", "open", "high", "low", "close", "wap",
        "volume", "bar_count", "n_valid_1m", "is_valid", "session_id",
        "session_phase", "bar_in_session", "is_roll_boundary",
    }
    assert bars.index.names == ["root", "timestamp"]
    assert required <= set(bars.columns)
    assert str(bars.index.get_level_values("timestamp").tz) == "UTC"
    ratios = bars.groupby(level="root")["is_valid"].mean()
    assert ratios["ES"] >= 0.70
    assert ratios["HSI"] >= 0.55
    assert int(bars["is_roll_boundary"].sum()) == 18
    assert bars.loc[bars["is_roll_boundary"], "ret"].isna().all()


def test_generated_cost_and_session_configs_are_locked() -> None:
    with (ROOT / "configs" / "cost_model.yaml").open(encoding="utf-8") as handle:
        costs = yaml.safe_load(handle)
    with (ROOT / "configs" / "session_defs.yaml").open(encoding="utf-8") as handle:
        sessions = yaml.safe_load(handle)["session_defs"]

    assert costs["cost_multiplier"] == 1.5
    hsi = [item for item in sessions if item["root"] == "HSI"]
    assert hsi[0]["segments"][0]["start"] == "17:15"
    assert hsi[1]["valid_from"] == "2026-07-20"
    assert hsi[1]["segments"][0]["start"] == "17:00"

