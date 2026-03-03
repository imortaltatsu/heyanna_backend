"""FastAPI application for real-time prediction market analysis.

This API provides real-time access to prediction market analysis data,
including win rates, returns, volume trends, and more. All analysis
is computed from Kalshi market and trade data.
"""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import duckdb
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tqdm import tqdm

from src.api.cache import AnalysisCache

app = FastAPI(
    title="Prediction Market Analysis API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_cache: Optional[AnalysisCache] = None

# Central list of analysis names used across endpoints and background workers.
ANALYSIS_NAMES: list[str] = [
    "win_rate_by_price",
    "returns_by_hour",
    "volume_over_time",
    "maker_taker_gap_over_time",
    "ev_yes_vs_no",
    "mispricing_by_price",
    "longshot_volume_share_over_time",
    "trade_size_by_role",
    "maker_vs_taker_returns",
    "maker_returns_by_direction",
    "maker_win_rate_by_direction",
    "yes_vs_no_by_price",
    "maker_taker_returns_by_category",
    "win_rate_by_trade_size",
    "kalshi_calibration_deviation_over_time",
    "meta_stats",
]


class TTLCache:
    """Simple in-memory TTL cache for fast, near-real-time endpoints."""

    def __init__(self, ttl_seconds: int = 10):
        self.ttl_seconds = ttl_seconds
        self._data: dict[tuple, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: tuple) -> Optional[Any]:
        now = time.time()
        with self._lock:
            item = self._data.get(key)
            if not item:
                return None
            expires_at, value = item
            if expires_at < now:
                del self._data[key]
                return None
            return value

    def set(self, key: tuple, value: Any) -> None:
        expires_at = time.time() + self.ttl_seconds
        with self._lock:
            self._data[key] = (expires_at, value)


_market_search_cache = TTLCache(ttl_seconds=2)
_market_detail_cache = TTLCache(ttl_seconds=2)
_market_trades_cache = TTLCache(ttl_seconds=1)


def get_base_dir() -> Path:
    """Get the base directory for data files."""
    possible_dirs = [
        Path.cwd(),
        Path(__file__).parent.parent.parent,
        Path("/workspace/proj/prediction-market-analysis"),
    ]
    for d in possible_dirs:
        if (d / "data" / "kalshi" / "trades").exists():
            return d
    return Path.cwd()


def get_cache_instance() -> AnalysisCache:
    """Get or create the cache instance."""
    global _cache
    if _cache is None:
        _cache = AnalysisCache(base_dir=get_base_dir())
    return _cache


class Market(BaseModel):
    """Kalshi market model."""
    ticker: str
    event_ticker: str
    market_type: str
    title: str
    yes_sub_title: str
    no_sub_title: str
    status: str
    yes_bid: Optional[int] = None
    yes_ask: Optional[int] = None
    no_bid: Optional[int] = None
    no_ask: Optional[int] = None
    last_price: Optional[int] = None
    volume: int = 0
    volume_24h: int = 0
    open_interest: int = 0
    result: str = ""
    created_time: Optional[datetime] = None
    open_time: Optional[datetime] = None
    close_time: Optional[datetime] = None


class Trade(BaseModel):
    """Kalshi trade model."""
    trade_id: str
    ticker: str
    count: int
    yes_price: int
    no_price: int
    taker_side: str
    created_time: datetime


class AnalysisResponse(BaseModel):
    """Response for analysis queries."""
    name: str
    description: str
    data: list[dict[str, Any]]
    chart: Optional[dict[str, Any]] = None
    refreshed_at: datetime


class DashboardResponse(BaseModel):
    """Complete dashboard response with all metrics."""
    summary: dict[str, Any]
    win_rate_by_price: dict[str, Any]
    returns_by_hour: dict[str, Any]
    volume_over_time: dict[str, Any]
    maker_taker_gap: dict[str, Any]
    ev_yes_vs_no: dict[str, Any]
    mispricing_by_price: dict[str, Any]
    longshot_volume_share: dict[str, Any]
    trade_size_by_role: dict[str, Any]
    maker_vs_taker_returns: dict[str, Any]
    maker_returns_by_direction: dict[str, Any]
    maker_win_rate_by_direction: dict[str, Any]
    yes_vs_no_by_price: dict[str, Any]
    maker_taker_returns_by_category: dict[str, Any]
    win_rate_by_trade_size: dict[str, Any]
    kalshi_calibration_deviation: dict[str, Any]
    refreshed_at: datetime
    pending_analyses: list[str] = []


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint."""
    base_dir = get_base_dir()
    trades_dir = base_dir / "data" / "kalshi" / "trades"
    markets_dir = base_dir / "data" / "kalshi" / "markets"

    # API is considered "ready" when core data dirs exist and have data
    trades_ok = trades_dir.exists() and any(trades_dir.glob("*.parquet"))
    markets_ok = markets_dir.exists() and any(markets_dir.glob("*.parquet"))

    cache_stats = get_cache_instance().get_stats()
    api_ready = trades_ok and markets_ok

    return {
        "status": "healthy",
        "api_ready": api_ready,
        "timestamp": datetime.utcnow().isoformat(),
        "data_sources": {
            "trades_dir": str(trades_dir),
            "trades_ok": trades_ok,
            "markets_dir": str(markets_dir),
            "markets_ok": markets_ok,
        },
        "cache_size": cache_stats["size"],
        "cache_refresh_interval_seconds": cache_stats["refresh_interval_seconds"],
    }


@app.get("/api/v1/dashboard")
async def get_dashboard() -> DashboardResponse:
    """Get complete dashboard data with all key metrics.

    This endpoint returns all analysis data in a single request,
    making it ideal for dashboard applications.
    """
    cache = get_cache_instance()

    # Only serve already-computed analyses. If anything is still computing in
    # the background, return a 202 with a clear status instead of blocking.
    entries = await asyncio.gather(
        cache.get_cached("meta_stats"),
        cache.get_cached("win_rate_by_price"),
        cache.get_cached("returns_by_hour"),
        cache.get_cached("volume_over_time"),
        cache.get_cached("maker_taker_gap_over_time"),
        cache.get_cached("ev_yes_vs_no"),
        cache.get_cached("mispricing_by_price"),
        cache.get_cached("longshot_volume_share_over_time"),
        cache.get_cached("trade_size_by_role"),
        cache.get_cached("maker_vs_taker_returns"),
        cache.get_cached("maker_returns_by_direction"),
        cache.get_cached("maker_win_rate_by_direction"),
        cache.get_cached("yes_vs_no_by_price"),
        cache.get_cached("maker_taker_returns_by_category"),
        cache.get_cached("win_rate_by_trade_size"),
        cache.get_cached("kalshi_calibration_deviation_over_time"),
    )

    (
        meta_entry,
        win_rate_entry,
        returns_entry,
        volume_entry,
        maker_taker_entry,
        ev_entry,
        mispricing_entry,
        longshot_entry,
        trade_size_entry,
        maker_vs_taker_entry,
        maker_direction_entry,
        maker_win_entry,
        yes_no_entry,
        category_entry,
        trade_size_win_entry,
        calibration_entry,
    ) = entries

    names_in_order = [
        "meta_stats",
        "win_rate_by_price",
        "returns_by_hour",
        "volume_over_time",
        "maker_taker_gap_over_time",
        "ev_yes_vs_no",
        "mispricing_by_price",
        "longshot_volume_share_over_time",
        "trade_size_by_role",
        "maker_vs_taker_returns",
        "maker_returns_by_direction",
        "maker_win_rate_by_direction",
        "yes_vs_no_by_price",
        "maker_taker_returns_by_category",
        "win_rate_by_trade_size",
        "kalshi_calibration_deviation_over_time",
    ]

    pending = [
        name for name, entry in zip(names_in_order, entries) if entry is None
    ]

    # If nothing is ready yet, tell the client it's still computing.
    ready_entries = [e for e in entries if e is not None]
    if not ready_entries:
        raise HTTPException(
            status_code=202,
            detail={
                "status": "computing",
                "message": "Dashboard analyses are still being computed and will be available shortly.",
                "pending_analyses": pending,
            },
        )

    meta_data = meta_entry.data.get("data", []) if meta_entry else []
    metrics = {row.get("metric"): row.get("value") for row in meta_data}

    min_created_at = min(e.created_at for e in ready_entries)

    return DashboardResponse(
        summary={
            "num_trades": int(metrics.get("num_trades", 0)),
            "num_trades_millions": metrics.get("num_trades_millions", 0),
            "total_volume_billions": metrics.get("total_volume_billions", 0),
            "num_markets": int(metrics.get("num_markets", 0)),
            "num_events": int(metrics.get("num_events", 0)),
        },
        win_rate_by_price={
            "data": win_rate_entry.data.get("data", []),
            "chart": win_rate_entry.data.get("chart"),
        },
        returns_by_hour={
            "data": returns_entry.data.get("data", []),
            "chart": returns_entry.data.get("chart"),
        },
        volume_over_time={
            "data": volume_entry.data.get("data", []),
            "chart": volume_entry.data.get("chart"),
        },
        maker_taker_gap={
            "data": maker_taker_entry.data.get("data", []),
            "chart": maker_taker_entry.data.get("chart"),
        },
        ev_yes_vs_no={
            "data": ev_entry.data.get("data", []),
            "chart": ev_entry.data.get("chart"),
        },
        mispricing_by_price={
            "data": mispricing_entry.data.get("data", []),
            "chart": mispricing_entry.data.get("chart"),
        },
        longshot_volume_share={
            "data": longshot_entry.data.get("data", []),
            "chart": longshot_entry.data.get("chart"),
        },
        trade_size_by_role={
            "data": trade_size_entry.data.get("data", []),
            "chart": trade_size_entry.data.get("chart"),
        },
        maker_vs_taker_returns={
            "data": maker_vs_taker_entry.data.get("data", []),
            "chart": maker_vs_taker_entry.data.get("chart"),
        },
        maker_returns_by_direction={
            "data": maker_direction_entry.data.get("data", []),
            "chart": maker_direction_entry.data.get("chart"),
        },
        maker_win_rate_by_direction={
            "data": maker_win_entry.data.get("data", []),
            "chart": maker_win_entry.data.get("chart"),
        },
        yes_vs_no_by_price={
            "data": yes_no_entry.data.get("data", []),
            "chart": yes_no_entry.data.get("chart"),
        },
        maker_taker_returns_by_category={
            "data": category_entry.data.get("data", []) if category_entry else [],
            "chart": category_entry.data.get("chart") if category_entry else None,
        },
        win_rate_by_trade_size={
            "data": trade_size_win_entry.data.get("data", []) if trade_size_win_entry else [],
            "chart": trade_size_win_entry.data.get("chart") if trade_size_win_entry else None,
        },
        kalshi_calibration_deviation={
            "data": calibration_entry.data.get("data", []) if calibration_entry else [],
            "chart": calibration_entry.data.get("chart") if calibration_entry else None,
        },
        refreshed_at=datetime.fromtimestamp(min_created_at),
        pending_analyses=pending,
    )


@app.get("/api/v1/analysis/win_rate_by_price")
async def get_win_rate_by_price() -> AnalysisResponse:
    """Get win rate analysis by price level.

    This analysis shows how actual outcomes compare to implied
    probabilities across different price levels. It helps detect
    market mispricing where the market's implied probability
    differs from the actual win rate.

    Expected use: Identify markets where the price doesn't reflect
    the true probability of the outcome.
    """
    cache = get_cache_instance()
    entry = await cache.get_cached("win_rate_by_price")
    if entry is None:
        raise HTTPException(
            status_code=202,
            detail={
                "status": "computing",
                "message": "win_rate_by_price analysis is still computing and will be available shortly.",
            },
        )
    return AnalysisResponse(
        name="win_rate_by_price",
        description="Win rate vs price market calibration",
        data=entry.data.get("data", []),
        chart=entry.data.get("chart"),
        refreshed_at=datetime.fromtimestamp(entry.created_at),
    )


@app.get("/api/v1/analysis/returns_by_hour")
async def get_returns_by_hour() -> AnalysisResponse:
    """Get excess returns by hour of day.

    This analysis shows when the market tends to have higher or lower
    returns. It helps identify temporal patterns in market behavior
    that may be exploitable.

    Expected use: Determine optimal trading hours based on historical
    return patterns.
    """
    cache = get_cache_instance()
    entry = await cache.get_cached("returns_by_hour")
    if entry is None:
        raise HTTPException(
            status_code=202,
            detail={
                "status": "computing",
                "message": "returns_by_hour analysis is still computing and will be available shortly.",
            },
        )
    return AnalysisResponse(
        name="returns_by_hour",
        description="Excess returns by hour of day (ET)",
        data=entry.data.get("data", []),
        chart=entry.data.get("chart"),
        refreshed_at=datetime.fromtimestamp(entry.created_at),
    )


@app.get("/api/v1/analysis/volume_over_time")
async def get_volume_over_time() -> AnalysisResponse:
    """Get volume over time.

    Tracks quarterly trading volume to understand market adoption
    and activity patterns. Helps identify growth trends and
    seasonal patterns.

    Expected use: Monitor market health and growth over time.
    """
    cache = get_cache_instance()
    entry = await cache.get_cached("volume_over_time")
    if entry is None:
        raise HTTPException(
            status_code=202,
            detail={
                "status": "computing",
                "message": "volume_over_time analysis is still computing and will be available shortly.",
            },
        )
    return AnalysisResponse(
        name="volume_over_time",
        description="Quarterly notional volume",
        data=entry.data.get("data", []),
        chart=entry.data.get("chart"),
        refreshed_at=datetime.fromtimestamp(entry.created_at),
    )


@app.get("/api/v1/analysis/maker_taker_gap")
async def get_maker_taker_gap() -> AnalysisResponse:
    """Get maker-taker gap analysis.

    Shows maker-taker excess returns over time, indicating who is
    profiting from market movements. A positive gap favors takers,
    negative favors makers.

    Expected use: Understand the competitive dynamics between
    market makers and takers.
    """
    cache = get_cache_instance()
    entry = await cache.get_cached("maker_taker_gap_over_time")
    if entry is None:
        raise HTTPException(
            status_code=202,
            detail={
                "status": "computing",
                "message": "maker_taker_gap_over_time analysis is still computing and will be available shortly.",
            },
        )
    return AnalysisResponse(
        name="maker_taker_gap_over_time",
        description="Quarterly maker-taker excess returns over time",
        data=entry.data.get("data", []),
        chart=entry.data.get("chart"),
        refreshed_at=datetime.fromtimestamp(entry.created_at),
    )


@app.get("/api/v1/analysis/ev_yes_vs_no")
async def get_ev_yes_vs_no() -> AnalysisResponse:
    """Get expected value comparison for YES vs NO bets.

    Compares the expected value of YES vs NO bets across different
    price levels. Helps identify which side of a bet has better
    expected value at various price points.

    Expected use: Identify favorable betting opportunities based
    on expected value analysis.
    """
    cache = get_cache_instance()
    entry = await cache.get_cached("ev_yes_vs_no")
    if entry is None:
        raise HTTPException(
            status_code=202,
            detail={
                "status": "computing",
                "message": "ev_yes_vs_no analysis is still computing and will be available shortly.",
            },
        )
    return AnalysisResponse(
        name="ev_yes_vs_no",
        description="Expected value comparison of YES vs NO bets by price level",
        data=entry.data.get("data", []),
        chart=entry.data.get("chart"),
        refreshed_at=datetime.fromtimestamp(entry.created_at),
    )


@app.get("/api/v1/analysis/mispricing_by_price")
async def get_mispricing_by_price() -> AnalysisResponse:
    """Get mispricing analysis by price.

    Analyzes how much the market price deviates from the true
    probability at different price levels. Helps identify
    arbitrage opportunities.

    Expected use: Find markets where the price doesn't reflect
    the true probability of the outcome.
    """
    cache = get_cache_instance()
    entry = await cache.get_cached("mispricing_by_price")
    if entry is None:
        raise HTTPException(
            status_code=202,
            detail={
                "status": "computing",
                "message": "mispricing_by_price analysis is still computing and will be available shortly.",
            },
        )
    return AnalysisResponse(
        name="mispricing_by_price",
        description="Mispricing analysis by contract price",
        data=entry.data.get("data", []),
        chart=entry.data.get("chart"),
        refreshed_at=datetime.fromtimestamp(entry.created_at),
    )


@app.get("/api/v1/analysis/longshot_volume_share")
async def get_longshot_volume_share() -> AnalysisResponse:
    """Get longshot volume share over time.

    Tracks the volume share of longshot contracts to understand
    risk appetite in the market. Higher longshot volume may
    indicate increased risk appetite.

    Expected use: Monitor market sentiment and risk appetite
    over time.
    """
    cache = get_cache_instance()
    entry = await cache.get_cached("longshot_volume_share_over_time")
    if entry is None:
        raise HTTPException(
            status_code=202,
            detail={
                "status": "computing",
                "message": "longshot_volume_share_over_time analysis is still computing and will be available shortly.",
            },
        )
    return AnalysisResponse(
        name="longshot_volume_share_over_time",
        description="Volume share of longshot contracts over time",
        data=entry.data.get("data", []),
        chart=entry.data.get("chart"),
        refreshed_at=datetime.fromtimestamp(entry.created_at),
    )


@app.get("/api/v1/analysis/trade_size_by_role")
async def get_trade_size_by_role() -> AnalysisResponse:
    """Get trade sizes by role (maker/taker).

    Analyzes how trade sizes differ between market makers and
    takers. Market makers typically place smaller orders to
    provide liquidity.

    Expected use: Understand market maker behavior and identify
    large institutional trades.
    """
    cache = get_cache_instance()
    entry = await cache.get_cached("trade_size_by_role")
    if entry is None:
        raise HTTPException(
            status_code=202,
            detail={
                "status": "computing",
                "message": "trade_size_by_role analysis is still computing and will be available shortly.",
            },
        )
    return AnalysisResponse(
        name="trade_size_by_role",
        description="Trade sizes by market role",
        data=entry.data.get("data", []),
        chart=entry.data.get("chart"),
        refreshed_at=datetime.fromtimestamp(entry.created_at),
    )


@app.get("/api/v1/analysis/maker_vs_taker_returns")
async def get_maker_vs_taker_returns() -> AnalysisResponse:
    """Get returns comparison between makers and takers.

    Shows which side has better returns on average. This can
    indicate whether the market favors liquidity providers
    or takers.

    Expected use: Determine whether to act as a market maker
    or taker based on historical returns.
    """
    cache = get_cache_instance()
    entry = await cache.get_cached("maker_vs_taker_returns")
    if entry is None:
        raise HTTPException(
            status_code=202,
            detail={
                "status": "computing",
                "message": "maker_vs_taker_returns analysis is still computing and will be available shortly.",
            },
        )
    return AnalysisResponse(
        name="maker_vs_taker_returns",
        description="Returns comparison between market makers and takers",
        data=entry.data.get("data", []),
        chart=entry.data.get("chart"),
        refreshed_at=datetime.fromtimestamp(entry.created_at),
    )


@app.get("/api/v1/analysis/maker_returns_by_direction")
async def get_maker_returns_by_direction() -> AnalysisResponse:
    """Get maker returns broken down by YES/NO direction.

    Shows whether market makers have different returns depending
    on which side they're on. Can reveal directional biases.

    Expected use: Understand maker behavior and identify
    directional patterns.
    """
    cache = get_cache_instance()
    entry = await cache.get_cached("maker_returns_by_direction")
    if entry is None:
        raise HTTPException(
            status_code=202,
            detail={
                "status": "computing",
                "message": "maker_returns_by_direction analysis is still computing and will be available shortly.",
            },
        )
    return AnalysisResponse(
        name="maker_returns_by_direction",
        description="Market maker returns by YES/NO direction",
        data=entry.data.get("data", []),
        chart=entry.data.get("chart"),
        refreshed_at=datetime.fromtimestamp(entry.created_at),
    )


@app.get("/api/v1/analysis/maker_win_rate_by_direction")
async def get_maker_win_rate_by_direction() -> AnalysisResponse:
    """Get maker win rates by YES/NO direction.

    Shows the win rates for market makers on each side of the
    market. Can reveal which direction is more profitable
    for liquidity providers.

    Expected use: Determine which side to provide liquidity on
    for optimal maker returns.
    """
    cache = get_cache_instance()
    entry = await cache.get_cached("maker_win_rate_by_direction")
    if entry is None:
        raise HTTPException(
            status_code=202,
            detail={
                "status": "computing",
                "message": "maker_win_rate_by_direction analysis is still computing and will be available shortly.",
            },
        )
    return AnalysisResponse(
        name="maker_win_rate_by_direction",
        description="Market maker win rates by YES/NO direction",
        data=entry.data.get("data", []),
        chart=entry.data.get("chart"),
        refreshed_at=datetime.fromtimestamp(entry.created_at),
    )


@app.get("/api/v1/analysis/yes_vs_no_by_price")
async def get_yes_vs_no_by_price() -> AnalysisResponse:
    """Get YES vs NO volume comparison by price level.

    Shows the balance of buying pressure between YES and NO
    contracts at different price points. Helps understand
    market sentiment.

    Expected use: Gauge market sentiment and balance between
    YES and NO positions.
    """
    cache = get_cache_instance()
    entry = await cache.get_cached("yes_vs_no_by_price")
    if entry is None:
        raise HTTPException(
            status_code=202,
            detail={
                "status": "computing",
                "message": "yes_vs_no_by_price analysis is still computing and will be available shortly.",
            },
        )
    return AnalysisResponse(
        name="yes_vs_no_by_price",
        description="YES vs NO volume by price level",
        data=entry.data.get("data", []),
        chart=entry.data.get("chart"),
        refreshed_at=datetime.fromtimestamp(entry.created_at),
    )


@app.get("/api/v1/analysis/maker_taker_returns_by_category")
async def get_maker_taker_returns_by_category() -> AnalysisResponse:
    """Get returns broken down by event category.

    Shows how different categories of events perform in terms
    of returns. Helps identify which event types are most
    profitable.

    Expected use: Identify which categories have the best
    returns for trading strategy.
    """
    cache = get_cache_instance()
    entry = await cache.get_cached("maker_taker_returns_by_category")
    if entry is None:
        raise HTTPException(
            status_code=202,
            detail={
                "status": "computing",
                "message": "maker_taker_returns_by_category analysis is still computing and will be available shortly.",
            },
        )
    return AnalysisResponse(
        name="maker_taker_returns_by_category",
        description="Returns by event category",
        data=entry.data.get("data", []),
        chart=entry.data.get("chart"),
        refreshed_at=datetime.fromtimestamp(entry.created_at),
    )


@app.get("/api/v1/analysis/win_rate_by_trade_size")
async def get_win_rate_by_trade_size() -> AnalysisResponse:
    """Get win rates by trade size.

    Analyzes how win rates vary by trade size. Large trades
    may have different success rates due to market impact.

    Expected use: Optimize trade sizes based on historical
    win rate performance.
    """
    cache = get_cache_instance()
    entry = await cache.get_cached("win_rate_by_trade_size")
    if entry is None:
        raise HTTPException(
            status_code=202,
            detail={
                "status": "computing",
                "message": "win_rate_by_trade_size analysis is still computing and will be available shortly.",
            },
        )
    return AnalysisResponse(
        name="win_rate_by_trade_size",
        description="Win rates by trade size",
        data=entry.data.get("data", []),
        chart=entry.data.get("chart"),
        refreshed_at=datetime.fromtimestamp(entry.created_at),
    )


@app.get("/api/v1/analysis/kalshi_calibration_deviation")
async def get_kalshi_calibration_deviation() -> AnalysisResponse:
    """Get calibration deviation over time.

    Tracks how well the market prices reflect actual outcomes
    over time. Calibration measures the accuracy of probability
    predictions.

    Expected use: Monitor market accuracy and identify periods
    of mispricing.
    """
    cache = get_cache_instance()
    entry = await cache.get_cached("kalshi_calibration_deviation_over_time")
    if entry is None:
        raise HTTPException(
            status_code=202,
            detail={
                "status": "computing",
                "message": "kalshi_calibration_deviation_over_time analysis is still computing and will be available shortly.",
            },
        )
    return AnalysisResponse(
        name="kalshi_calibration_deviation_over_time",
        description="Calibration deviation over time",
        data=entry.data.get("data", []),
        chart=entry.data.get("chart"),
        refreshed_at=datetime.fromtimestamp(entry.created_at),
    )


@app.get("/api/v1/analysis/{analysis_name}")
async def get_analysis(analysis_name: str) -> AnalysisResponse:
    """Get any analysis by name.

    Valid analysis names:
    - win_rate_by_price
    - returns_by_hour
    - volume_over_time
    - maker_taker_gap_over_time
    - ev_yes_vs_no
    - mispricing_by_price
    - longshot_volume_share_over_time
    - trade_size_by_role
    - maker_vs_taker_returns
    - maker_returns_by_direction
    - maker_win_rate_by_direction
    - yes_vs_no_by_price
    - maker_taker_returns_by_category
    - win_rate_by_trade_size
    - kalshi_calibration_deviation_over_time
    - meta_stats
    """
    cache = get_cache_instance()

    if analysis_name not in ANALYSIS_NAMES:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown analysis. Valid options: {', '.join(ANALYSIS_NAMES)}",
        )
    entry = await cache.get_cached(analysis_name)
    if entry is None:
        raise HTTPException(
            status_code=202,
            detail={
                "status": "computing",
                "message": f"{analysis_name} analysis is still computing and will be available shortly.",
            },
        )
    return AnalysisResponse(
        name=analysis_name,
        description=entry.data.get("description", ""),
        data=entry.data.get("data", []),
        chart=entry.data.get("chart"),
        refreshed_at=datetime.fromtimestamp(entry.created_at),
    )


@app.get("/api/v1/market/search")
async def search_markets(
    query: str = Query(..., description="Search query for market title or ticker"),
    limit: int = Query(10, le=100, description="Maximum results"),
) -> dict[str, Any]:
    """Search for markets by query."""
    cache_key = (query, limit)
    cached = _market_search_cache.get(cache_key)
    if cached is not None:
        return cached

    base_dir = get_base_dir()
    start_time = time.time()

    async def _run_query() -> list[tuple]:
        def _query() -> list[tuple]:
            con = duckdb.connect()
            try:
                return con.execute(
                    f"""
                    SELECT
                        ticker,
                        event_ticker,
                        market_type,
                        title,
                        yes_sub_title,
                        no_sub_title,
                        status,
                        yes_bid,
                        yes_ask,
                        no_bid,
                        no_ask,
                        last_price,
                        volume,
                        volume_24h,
                        open_interest,
                        result,
                        created_time,
                        open_time,
                        close_time
                    FROM '{base_dir / "data" / "kalshi" / "markets" / "*.parquet"}'
                    WHERE LOWER(title) LIKE LOWER('%{query}%')
                       OR LOWER(ticker) LIKE LOWER('%{query}%')
                    LIMIT {limit}
                    """
                ).fetchall()
            finally:
                con.close()

        return await asyncio.to_thread(_query)

    results = await _run_query()

    columns = [
        "ticker", "event_ticker", "market_type", "title", "yes_sub_title",
        "no_sub_title", "status", "yes_bid", "yes_ask", "no_bid", "no_ask",
        "last_price", "volume", "volume_24h", "open_interest", "result",
        "created_time", "open_time", "close_time"
    ]

    markets = [Market(**dict(zip(columns, row))) for row in results]

    response = {
        "results": [m.model_dump() for m in markets],
        "total": len(markets),
        "query": query,
        "time_ms": int((time.time() - start_time) * 1000),
    }
    _market_search_cache.set(cache_key, response)
    return response


@app.get("/api/v1/market/{ticker}")
async def get_market(ticker: str) -> Market:
    """Get detailed information for a specific market."""
    cache_key = (ticker,)
    cached = _market_detail_cache.get(cache_key)
    if cached is not None:
        return cached

    base_dir = get_base_dir()

    async def _run_query() -> Optional[tuple]:
        def _query() -> Optional[tuple]:
            con = duckdb.connect()
            try:
                return con.execute(
                    f"""
                    SELECT
                        ticker,
                        event_ticker,
                        market_type,
                        title,
                        yes_sub_title,
                        no_sub_title,
                        status,
                        yes_bid,
                        yes_ask,
                        no_bid,
                        no_ask,
                        last_price,
                        volume,
                        volume_24h,
                        open_interest,
                        result,
                        created_time,
                        open_time,
                        close_time
                    FROM '{base_dir / "data" / "kalshi" / "markets" / "*.parquet"}'
                    WHERE ticker = ?
                    LIMIT 1
                    """,
                    [ticker],
                ).fetchone()
            finally:
                con.close()

        return await asyncio.to_thread(_query)

    result = await _run_query()

    if not result:
        raise HTTPException(status_code=404, detail=f"Market {ticker} not found")

    columns = [
        "ticker", "event_ticker", "market_type", "title", "yes_sub_title",
        "no_sub_title", "status", "yes_bid", "yes_ask", "no_bid", "no_ask",
        "last_price", "volume", "volume_24h", "open_interest", "result",
        "created_time", "open_time", "close_time"
    ]

    market = Market(**dict(zip(columns, result)))
    _market_detail_cache.set(cache_key, market)
    return market


@app.get("/api/v1/market/{ticker}/trades")
async def get_market_trades(
    ticker: str,
    limit: int = Query(100, le=10000, description="Maximum trades to return"),
) -> list[Trade]:
    """Get recent trades for a specific market."""
    cache_key = (ticker, limit)
    cached = _market_trades_cache.get(cache_key)
    if cached is not None:
        return cached

    base_dir = get_base_dir()

    async def _run_query() -> list[tuple]:
        def _query() -> list[tuple]:
            con = duckdb.connect()
            try:
                return con.execute(
                    f"""
                    SELECT
                        trade_id,
                        ticker,
                        count,
                        yes_price,
                        no_price,
                        taker_side,
                        created_time
                    FROM '{base_dir / "data" / "kalshi" / "trades" / "*.parquet"}'
                    WHERE ticker = ?
                    ORDER BY created_time DESC
                    LIMIT {limit}
                    """,
                    [ticker],
                ).fetchall()
            finally:
                con.close()

        return await asyncio.to_thread(_query)

    trades = await _run_query()

    columns = [
        "trade_id", "ticker", "count", "yes_price", "no_price",
        "taker_side", "created_time"
    ]

    models = [Trade(**dict(zip(columns, row))) for row in trades]
    _market_trades_cache.set(cache_key, models)
    return models


@app.post("/api/v1/refresh")
async def refresh_cache() -> dict[str, str]:
    """Force refresh all cached analysis.

    This endpoint clears the cache and triggers a refresh on the
    next request. Useful when new data has been indexed.
    """
    cache = get_cache_instance()
    await cache.refresh_all()
    return {
        "status": "refreshed",
        "message": "Cache has been cleared and will be repopulated on next request",
    }


# Background tasks for continuous indexing and analysis warmup
_background_task: Optional[asyncio.Task] = None
_analysis_task: Optional[asyncio.Task] = None


async def background_indexer(cache: AnalysisCache) -> None:
    """Continuously index new data and refresh cache."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    from src.indexers.kalshi.trades import KalshiTradesIndexer
    from src.indexers.kalshi.markets import KalshiMarketsIndexer

    # Use a conservative worker count in the always-on job to respect API rate limits.
    trade_indexer = KalshiTradesIndexer(max_workers=4)
    market_indexer = KalshiMarketsIndexer()

    # Use a thread pool for blocking I/O operations
    executor = ThreadPoolExecutor(max_workers=4)

    while True:
        try:
            # Index new trades in background thread
            print("Indexing new trades...")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(executor, trade_indexer.run)

            # Index new markets in background thread
            print("Indexing new markets...")
            await loop.run_in_executor(executor, market_indexer.run)

            # Refresh all analysis
            print("Refreshing all analysis...")
            await cache.refresh_all()

            print("Background indexing complete. Next run in 5 minutes.")
            await asyncio.sleep(300)  # 5 minutes
        except Exception as e:
            print(f"Background indexer error: {e}")
            await asyncio.sleep(60)  # Wait 1 minute on error


async def background_analysis_worker(cache: AnalysisCache) -> None:
    """Continuously precompute analysis results so APIs stay fast."""
    while True:
        try:
            print("Warming analysis cache...")
            # Warm all analyses with a tqdm progress bar; heavy work runs in
            # background threads via AnalysisCache._compute_analysis.
            # Run several analyses concurrently to better use CPU while
            # respecting DuckDB and I/O limits on a 32-core machine.
            concurrency = 8
            pbar = tqdm(total=len(ANALYSIS_NAMES), desc="Analyses", leave=False)

            sem = asyncio.Semaphore(concurrency)

            async def _warm(name: str) -> None:
                async with sem:
                    try:
                        await cache.get_or_refresh(name)
                    except Exception as e:
                        print(f"Error warming analysis '{name}': {e}")
                    finally:
                        pbar.update(1)

            tasks = [asyncio.create_task(_warm(name)) for name in ANALYSIS_NAMES]
            await asyncio.gather(*tasks)
            pbar.close()

            print("Analysis cache warmup complete. Next run in 5 minutes.")
            await asyncio.sleep(300)  # 5 minutes
        except Exception as e:
            print(f"Background analysis worker error: {e}")
            await asyncio.sleep(60)


@app.on_event("startup")
async def start_background_indexer() -> None:
    """Start background indexer on startup."""
    global _background_task, _analysis_task
    cache = get_cache_instance()
    _background_task = asyncio.create_task(background_indexer(cache))
    _analysis_task = asyncio.create_task(background_analysis_worker(cache))


@app.on_event("shutdown")
async def stop_background_indexer() -> None:
    """Stop background indexer on shutdown."""
    global _background_task, _analysis_task
    if _background_task:
        _background_task.cancel()
        try:
            await _background_task
        except asyncio.CancelledError:
            pass
    if _analysis_task:
        _analysis_task.cancel()
        try:
            await _analysis_task
        except asyncio.CancelledError:
            pass