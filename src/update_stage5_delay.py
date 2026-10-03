"""Targeted repair/recalculation of the Stage 5 delay sensitivity."""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from src.backtest import aggregate_portfolio, run_event_backtest
from src.run_stage5 import FIGS, K_BASE, PORTFOLIOS, PROCESSED, ROOT, TABLES, TRADABLE, _load, _strategy_metrics


def main() -> None:
    bars, context, _, _, config, _ = _load()
    bars = bars.loc[bars.index.get_level_values("root").isin(TRADABLE)]
    context = context.reindex(bars.index)
    for name in PORTFOLIOS:
        sized = pd.read_parquet(PROCESSED / f"sizing_stage5_{name}.parquet").reindex(bars.index)
        sized["round_trip_cost_bp"] = context["round_trip_cost_bp"]
        rows = []
        for extra_delay in (0, 1, 2, 3):
            bt = run_event_backtest(
                bars, sized, execution_price="wap", cost_multiplier=1.5,
                delay_bars=1 + extra_delay,
                avoid_session_edge_bars=config["execution"]["avoid_session_edge_bars"],
                avoid_roll_boundary_bars=config["execution"]["avoid_roll_boundary_bars"],
                max_volume_participation=config["execution"]["max_volume_participation"],
            )
            _, daily = aggregate_portfolio(bt)
            rows.append(
                {"dimension": "delay", "setting": str(extra_delay),
                 **_strategy_metrics(bt, daily, K_BASE)}
            )
        path = TABLES / f"05_{name}_sensitivity.csv"
        table = pd.read_csv(path)
        table = pd.concat([table.loc[table.dimension != "delay"], pd.DataFrame(rows)], ignore_index=True)
        table.to_csv(path, index=False)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, dimension in zip(axes, ("cost_multiplier", "delay", "band_zeta")):
            frame = table.loc[table.dimension == dimension]
            ax.plot(frame.setting.astype(str), frame.sharpe, marker="o")
            ax.set(title=dimension, ylabel="Sharpe")
        fig.suptitle(f"Stage 5 {name}: Core sensitivity panel")
        fig.tight_layout()
        fig.savefig(FIGS / f"05_sensitivity_{name}.png", dpi=150)
        plt.close(fig)
        print(name)
        print(pd.DataFrame(rows)[["setting", "sharpe", "annualized_return", "executions"]].to_string(index=False))


if __name__ == "__main__":
    main()
