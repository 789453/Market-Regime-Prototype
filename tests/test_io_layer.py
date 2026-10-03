from pathlib import Path

import pandas as pd
import pytest

from src.io_layer import EXPECTED_COLUMNS, DataContractError, load


DATA_PATH = Path(
    r"D:\Trading\HongKongQuant\time_selecting_strategy"
    r"\raw_data\six_month_index_futures_minute(1).parquet"
)


def test_load_raw_data_meets_g0_contract() -> None:
    bars = load(DATA_PATH)

    assert bars.shape == (494_035, 12)
    assert tuple(bars.columns) == EXPECTED_COLUMNS
    assert bars["root"].nunique() == 5
    assert bars["local_symbol"].nunique() == 23
    assert bars.isna().sum().sum() == 0
    assert not bars.duplicated(["root", "timestamp"]).any()
    assert isinstance(bars["timestamp"].dtype, pd.DatetimeTZDtype)
    assert str(bars["timestamp"].dt.tz) == "UTC"


def test_root_filter_preserves_contract() -> None:
    bars = load(DATA_PATH, roots=["ES", "NQ"])

    assert set(bars["root"]) == {"ES", "NQ"}
    assert len(bars) == 272_174


def test_rejects_unknown_root() -> None:
    with pytest.raises(DataContractError, match="Unknown roots"):
        load(DATA_PATH, roots=["UNKNOWN"])

