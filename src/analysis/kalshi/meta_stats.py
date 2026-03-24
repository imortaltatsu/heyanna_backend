"""Generate meta statistics for the dataset."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from src.common.analysis import Analysis, AnalysisOutput


class MetaStatsAnalysis(Analysis):
    """Generate meta statistics for the Kalshi dataset."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="meta_stats",
            description="Dataset meta statistics including trade and market counts",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")

    def run(self) -> AnalysisOutput:
        """Execute the analysis and return outputs."""
        con = duckdb.connect()

        # Trade statistics
        trade_stats = con.execute(
            f"""
            SELECT
                COUNT(*) AS num_trades,
                SUM(count) AS total_volume,
                COUNT(DISTINCT ticker) AS num_tickers
            FROM '{self.trades_dir}/*.parquet'
            """
        ).fetchone()

        num_trades = trade_stats[0]
        total_volume = trade_stats[1]
        num_tickers_from_trades = trade_stats[2]

        # Market statistics
        market_stats = con.execute(
            f"""
            SELECT
                COUNT(*) AS num_markets,
                COUNT(DISTINCT event_ticker) AS num_events
            FROM '{self.markets_dir}/*.parquet'
            """
        ).fetchone()

        num_markets = market_stats[0]
        num_events = market_stats[1]

        # Global win rate across all positions (taker and maker sides).
        # This reuses the same resolved_markets + trade_positions pattern as
        # other analyses but collapses everything into a single snapshot
        # number so it can be served cheaply from cache.
        win_stats = con.execute(
            f"""
            WITH resolved_markets AS (
                SELECT ticker, result
                FROM '{self.markets_dir}/*.parquet'
                WHERE result IN ('yes', 'no')
            ),
            trade_positions AS (
                -- Buyer side (taker)
                SELECT
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
            )
            SELECT
                SUM(won) AS total_wins,
                COUNT(*) AS total_positions
            FROM trade_positions
            """
        ).fetchone()

        total_wins = win_stats[0] or 0
        total_positions = win_stats[1] or 0
        win_rate = (total_wins / total_positions) if total_positions > 0 else 0.0

        # Build DataFrame with statistics
        df = pd.DataFrame(
            [
                {
                    "metric": "num_trades",
                    "value": num_trades,
                    "formatted": self._format_number(num_trades),
                },
                {
                    "metric": "num_trades_millions",
                    "value": num_trades / 1e6,
                    "formatted": self._format_millions(num_trades),
                },
                {
                    "metric": "total_volume",
                    "value": total_volume,
                    "formatted": self._format_number(int(total_volume)),
                },
                {
                    "metric": "total_volume_billions",
                    "value": total_volume / 1e9,
                    "formatted": self._format_billions(total_volume),
                },
                {
                    "metric": "num_markets",
                    "value": num_markets,
                    "formatted": self._format_number(num_markets),
                },
                {
                    "metric": "num_events",
                    "value": num_events,
                    "formatted": self._format_number(num_events),
                },
                {
                    "metric": "num_tickers_from_trades",
                    "value": num_tickers_from_trades,
                    "formatted": self._format_number(num_tickers_from_trades),
                },
                {
                    "metric": "win_rate",
                    "value": win_rate,
                    "formatted": f"{win_rate:.4f}",
                },
                {
                    "metric": "win_rate_percent",
                    "value": win_rate * 100.0,
                    "formatted": f"{win_rate * 100.0:.2f}",
                },
            ]
        )

        # No figure or chart for this analysis (it generates data/stats only)
        return AnalysisOutput(figure=None, data=df, chart=None)

    @staticmethod
    def _format_number(n: int) -> str:
        """Format a number with commas."""
        return f"{n:,}"

    @staticmethod
    def _format_billions(n: float) -> str:
        """Format a number as billions with 2 decimal places."""
        return f"{n / 1e9:.2f}"

    @staticmethod
    def _format_millions(n: float) -> str:
        """Format a number as millions with 1 decimal place."""
        return f"{n / 1e6:.1f}"

    def generate_latex_macros(self) -> str:
        """Generate LaTeX macros for the statistics.

        Returns:
            String containing LaTeX macro definitions.
        """
        output = self.run()
        df = output.data

        # Convert to dict for easy lookup
        stats = {row["metric"]: row for _, row in df.iterrows()}

        tex_content = f"""% Auto-generated by MetaStatsAnalysis - do not edit manually
\\newcommand{{\\numTrades}}{{{stats["num_trades"]["formatted"]}}}
\\newcommand{{\\numTradesMillions}}{{{stats["num_trades_millions"]["formatted"]}}}
\\newcommand{{\\totalVolume}}{{{stats["total_volume"]["formatted"]}}}
\\newcommand{{\\totalVolumeBillions}}{{{stats["total_volume_billions"]["formatted"]}}}
\\newcommand{{\\numMarkets}}{{{stats["num_markets"]["formatted"]}}}
\\newcommand{{\\numEvents}}{{{stats["num_events"]["formatted"]}}}
"""
        return tex_content
