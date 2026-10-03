"""Register every coordinate of the predictive-state experiment."""

from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs/crypto_predictive_states.yaml").read_text(encoding="utf-8"))
OUT = ROOT / "reports/crypto/predictive_states_v1/feature_registry.csv"

DEFINITIONS = {
    "rv_fast_slow": ("(RV_fast / fast) / (RV_slow / slow)", "fast and slow", "future risk may rise; return sign unknown"),
    "rv_surprise": ("RV_fast / rolling_mean(RV_fast, slow)", "fast within slow", "future risk may rise; return sign unknown"),
    "semivar_balance": ("(up_semivar_fast - down_semivar_fast) / (up + down)", "fast", "direction may persist or mean revert"),
    "lower_wick_med": ("rolling_mean((min(open,close)-low)/(high-low), med)", "med", "rejection/recovery candidate; sign unassigned"),
    "close_location": ("2*(close-low)/(high-low)-1", "current bar", "upper close suggests positive contemporaneous path; future unknown"),
    "vwap_distance": ("log(close / rolling_VWAP_med)", "med", "overextension or continuation; future sign unassigned"),
    "momentum_z_med": ("log(close/close_lag_med)/sqrt(RV_med)", "med", "directional momentum candidate; future sign unassigned"),
    "direction_alignment": ("mean(sign(momentum_fast),sign(momentum_med),sign(momentum_slow))", "fast/med/slow", "contemporaneous direction; future unknown"),
    "efficiency_med": ("abs(momentum_med)/rolling_sum(abs(return), med)", "med", "high value means straighter path; future sign unknown"),
    "sign_flip_med": ("rolling_mean(sign(return_t)*sign(return_t-1)<0, med)", "med", "high value means more reversals; future sign unknown"),
    "flow_imb_med": ("2*sum(taker_buy_quote)/sum(quote_volume)-1", "med", "active-buy imbalance; future sign unknown"),
    "quote_rel_fast": ("mean(quote_volume, fast)/mean(quote_volume, slow)", "fast/slow", "activity change; future risk may rise"),
    "hour_sin": ("sin(2*pi*UTC_decimal_hour/24)", "current timestamp", "periodic, no monotone sign"),
    "hour_cos": ("cos(2*pi*UTC_decimal_hour/24)", "current timestamp", "periodic, no monotone sign"),
    "week_sin": ("sin(2*pi*UTC_decimal_weekday/7)", "current timestamp", "periodic, no monotone sign"),
    "week_cos": ("cos(2*pi*UTC_decimal_weekday/7)", "current timestamp", "periodic, no monotone sign"),
}


def main() -> None:
    rows = []
    for group, names in CFG["representation_columns"].items():
        for name in names:
            base, freq = name.rsplit("_", 1)
            formula, window, direction = DEFINITIONS[base]
            rows.append({"coordinate": name, "layer": "X0/X1/X2 current", "group": group,
                         "formula": formula, "window": f"{window}; {freq} bars (5m 12/48/288, 15m 4/16/96)",
                         "unit": "dimensionless before training percentile", "direction_expectation": direction,
                         "potential_redundancy": "same field other frequency and correlated group members",
                         "missing_behavior": "training percentile transform maps missing to 0.5; counted in missing_count; first 16 rows excluded",
                         "available_at": "close_time + 1ms of completed current bar"})
    for group, names in CFG["path_columns"].items():
        for name in names:
            for lag in (4, 8, 12, 16):
                rows.append({"coordinate": f"{name}__lag{lag}", "layer": "X1 lag16; X2 all lags", "group": group,
                             "formula": f"T_symbol,train({name}(t-{lag}*15m))",
                             "window": f"{lag * 15} minutes historical", "unit": "training percentile [0,1] before group scaling",
                             "direction_expectation": "none; learned from outcome distribution",
                             "potential_redundancy": "current coordinate and neighboring path lags",
                             "missing_behavior": "0.5 until lag exists; first 16 per symbol excluded",
                             "available_at": "historical bar was complete before current available_at"})
            for extreme in ("argmax", "argmin"):
                rows.append({"coordinate": f"{name}__{extreme}_order", "layer": "X2 ordered path", "group": group,
                             "formula": f"{extreme} of training-percentile values at t-4h,-3h,-2h,-1h,t, divided by 4",
                             "window": "4h; 5 ordered nodes", "unit": "node position [0,1] before group scaling",
                             "direction_expectation": "recent vs old peak/trough may alter transition; sign unassigned",
                             "potential_redundancy": "all five node levels; these positions summarize their ordering",
                             "missing_behavior": "inherits mapped values; first 16 per symbol excluded",
                             "available_at": "current completed bar available_at"})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"registered {len(rows)} coordinates at {OUT}")


if __name__ == "__main__":
    main()
