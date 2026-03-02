"""Pydantic models for API requests and responses."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel


class Market(BaseModel):
    """Market model from Kalshi data."""
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
    """Trade model from Kalshi data."""
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
    metadata: Optional[dict[str, Any]] = None
    refreshed_at: datetime


class DashboardStats(BaseModel):
    """Dashboard summary statistics."""
    num_trades: int
    num_trades_millions: float
    total_volume: int
    total_volume_billions: float
    num_markets: int
    num_events: int
    num_tickers: int
    last_updated: datetime


class SearchQuery(BaseModel):
    """Query for searching markets."""
    query: str
    limit: int = 10
    status: Optional[str] = None
    event_ticker: Optional[str] = None


class SearchResponse(BaseModel):
    """Response for market search."""
    results: list[Market]
    total: int
    query: str
    time_ms: int