"""Plot Kalshi calibration deviation over time."""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


class KalshiCalibrationDeviationOverTimeAnalysis(Analysis):
    """Analyze cumulative calibration deviation over time on Kalshi."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="kalshi_calibration_deviation_over_time",
            description="Kalshi calibration accuracy measured as mean absolute deviation over time",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")

    def run(self) -> AnalysisOutput:
        """Execute the analysis and return outputs."""
        con = duckdb.connect()

        # Compute weekly calibration deviation directly in DuckDB in a single
        # pass instead of looping in Python over each week. This is much
        # faster and scales better.
        output_df = con.execute(
            f"""
            WITH resolved_markets AS (
                SELECT ticker, result
                FROM '{self.markets_dir}/*.parquet'
                WHERE result IN ('yes', 'no')
            ),
            trade_positions AS (
                -- Buyer side (taker)
                SELECT
                    t.created_time,
                    CASE
                        WHEN t.taker_side = 'yes' THEN t.yes_price
                        ELSE t.no_price
                    END AS price,
                    CASE
                        WHEN t.taker_side = m.result THEN 1
                        ELSE 0
                    END AS won
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker
                WHERE t.yes_price > 0 AND t.no_price > 0

                UNION ALL

                -- Seller side (counterparty)
                SELECT
                    t.created_time,
                    CASE
                        WHEN t.taker_side = 'yes' THEN t.no_price
                        ELSE t.yes_price
                    END AS price,
                    CASE
                        WHEN t.taker_side != m.result THEN 1
                        ELSE 0
                    END AS won
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker
                WHERE t.yes_price > 0 AND t.no_price > 0
            ),
            per_price_week AS (
                SELECT
                    date_trunc('week', created_time) AS week,
                    price,
                    SUM(won) AS wins,
                    COUNT(*) AS total
                FROM trade_positions
                WHERE price BETWEEN 1 AND 99
                GROUP BY week, price
            ),
            per_week AS (
                SELECT
                    week AS date,
                    AVG(ABS(100.0 * wins / total - price)) AS mean_absolute_deviation
                FROM per_price_week
                GROUP BY week
                ORDER BY week
            )
            SELECT date, mean_absolute_deviation
            FROM per_week
            """
        ).df()

        # The API only needs tabular data and chart config. Creating full
        # matplotlib figures is relatively expensive and unnecessary for
        # the web dashboard, so we skip figure generation here to keep
        # responses fast and robust.
        chart = self._create_chart(output_df)

        return AnalysisOutput(figure=None, data=output_df, chart=chart)

    def _create_figure(self, dates: list, deviations: list) -> plt.Figure:
        """Create the matplotlib figure."""
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(dates, deviations, color="#4C72B0", linewidth=2)
        ax.fill_between(dates, deviations, alpha=0.3, color="#4C72B0")

        ax.set_xlabel("Date")
        ax.set_ylabel("Mean Absolute Deviation (%)")
        ax.set_title("Kalshi: Calibration Accuracy Over Time")

        ax.axhline(
            y=0,
            color="#D65F5F",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label="Perfect calibration",
        )

        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()

        plt.tight_layout()
        return fig

    def _create_chart(self, output_df: pd.DataFrame) -> ChartConfig:
        """Create the chart configuration for web display."""
        chart_data = [
            {
                "date": row["date"].strftime("%Y-%m-%d"),
                "deviation": round(row["mean_absolute_deviation"], 2),
            }
            for _, row in output_df.iterrows()
        ]

        return ChartConfig(
            type=ChartType.AREA,
            data=chart_data,
            xKey="date",
            yKeys=["deviation"],
            title="Kalshi: Calibration Accuracy Over Time",
            yUnit=UnitType.PERCENT,
            xLabel="Date",
            yLabel="Mean Absolute Deviation (%)",
        )
