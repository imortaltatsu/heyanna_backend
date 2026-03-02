"""Indexer for Kalshi trades data."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
from tqdm import tqdm

from src.common.indexer import Indexer
from src.indexers.kalshi.client import KalshiClient

DATA_DIR = Path("data/kalshi/trades")
MARKETS_DIR = Path("data/kalshi/markets")
CURSOR_FILE = Path("data/kalshi/.backfill_trades_cursor")


class KalshiTradesIndexer(Indexer):
    """Fetches and stores Kalshi trades data."""

    def __init__(
        self,
        min_ts: Optional[int] = None,
        max_ts: Optional[int] = None,
        max_workers: int = 8,
    ):
        super().__init__(
            name="kalshi_trades",
            description="Backfills Kalshi trades data to parquet files",
        )
        self._min_ts = min_ts
        self._max_ts = max_ts
        # Cap concurrent tickers to avoid hammering Kalshi's API.
        # This still uses multiple cores but stays within rate limits.
        self._num_workers = max(1, min(16, max_workers))

    def run(self) -> None:
        """Run the indexer to fetch and store trades data."""
        BATCH_SIZE = 10000
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        CURSOR_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Track which tickers have already been fully processed
        existing_tickers: set[str] = set()

        # Use cursor file to track progress instead of loading all existing trades
        cursor = None
        if CURSOR_FILE.exists():
            try:
                cursor = CURSOR_FILE.read_text().strip() or None
                if cursor:
                    print(f"Resuming from cursor: {cursor[:20]}...")
                    existing_tickers = set(cursor.split(",") if "," in cursor else [cursor])
            except Exception:
                pass

        # If no cursor, find existing parquet files for deduplication
        parquet_files = list(DATA_DIR.glob("trades_*.parquet"))
        if parquet_files and not cursor:
            print("Loading existing trades for deduplication...")
            try:
                result = duckdb.sql(
                    f"SELECT DISTINCT ticker FROM '{DATA_DIR}/trades_*.parquet'"
                ).fetchall()
                existing_tickers = {row[0] for row in result}
                print(f"Found {len(existing_tickers)} existing tickers with trades")
            except Exception:
                pass

        all_tickers = duckdb.sql(
            f"""
            SELECT DISTINCT ticker FROM '{MARKETS_DIR}/markets_*_*.parquet'
            WHERE volume >= 100
            ORDER BY ticker
        """
        ).fetchall()
        all_tickers = [row[0] for row in all_tickers]
        print(f"Found {len(all_tickers)} unique markets")

        # Filter to tickers not fully processed
        tickers_to_process = [t for t in all_tickers if t not in existing_tickers]
        print(
            f"Skipped {len(all_tickers) - len(tickers_to_process)} already processed, "
            f"{len(tickers_to_process)} to fetch"
        )

        if not tickers_to_process:
            print("Nothing to process")
            return

        all_trades: list[dict] = []
        total_trades_saved = 0
        next_chunk_idx = 0

        # Calculate next chunk index
        if parquet_files:
            indices: list[int] = []
            for f in parquet_files:
                parts = f.stem.split("_")
                if len(parts) >= 2:
                    try:
                        indices.append(int(parts[1]))
                    except ValueError:
                        continue
            if indices:
                next_chunk_idx = max(indices) + BATCH_SIZE

        def save_batch(trades_batch: list[dict]) -> int:
            nonlocal next_chunk_idx
            if not trades_batch:
                return 0
            chunk_path = DATA_DIR / f"trades_{next_chunk_idx}_{next_chunk_idx + BATCH_SIZE}.parquet"
            df = pd.DataFrame(trades_batch)
            df.to_parquet(chunk_path)
            next_chunk_idx += BATCH_SIZE
            return len(trades_batch)

        def fetch_ticker_trades(ticker: str) -> list[dict]:
            """Fetch trades for a single ticker."""
            client = KalshiClient()
            try:
                trades = client.get_market_trades(
                    ticker,
                    verbose=False,
                    min_ts=self._min_ts,
                    max_ts=self._max_ts,
                )
                if not trades:
                    return []
                fetched_at = datetime.utcnow()
                return [{**asdict(t), "_fetched_at": fetched_at} for t in trades]
            finally:
                client.close()

        # Concurrent fetching with a thread pool; writes happen on the main thread
        pbar = tqdm(total=len(tickers_to_process), desc="Fetching trades")

        def fetch_one(ticker: str) -> tuple[str, list[dict]]:
            try:
                trades_data = fetch_ticker_trades(ticker)
                return ticker, trades_data
            except Exception:
                return ticker, []

        with ThreadPoolExecutor(max_workers=self._num_workers) as executor:
            futures = {
                executor.submit(fetch_one, ticker): ticker for ticker in tickers_to_process
            }

            for future in as_completed(futures):
                ticker, trades_data = future.result()
                if trades_data:
                    all_trades.extend(trades_data)
                    # Flush in chunks to keep memory bounded
                    while len(all_trades) >= BATCH_SIZE:
                        batch = all_trades[:BATCH_SIZE]
                        total_trades_saved += save_batch(batch)
                        all_trades = all_trades[BATCH_SIZE:]

                pbar.update(1)
                pbar.set_postfix(saved=total_trades_saved, last=ticker[-20:])

        pbar.close()

        # Save remaining
        if all_trades:
            total_trades_saved += save_batch(all_trades)

        print(
            f"\nBackfill trades complete: {len(tickers_to_process)} markets processed, "
            f"{total_trades_saved} trades saved"
        )