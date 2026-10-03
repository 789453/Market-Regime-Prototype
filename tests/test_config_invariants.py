from src.io_layer import base_config, load_yaml, project_root, raw_data_path


def test_cost_multiplier_is_locked_at_one_point_five() -> None:
    root = project_root()
    assert base_config()["cost_multiplier"] == 1.5
    assert load_yaml(root / "configs" / "backtest.yaml")["execution"]["cost_multiplier"] == 1.5


def test_configured_raw_input_exists() -> None:
    assert raw_data_path().is_file()

