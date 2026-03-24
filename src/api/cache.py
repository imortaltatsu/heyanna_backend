"""Real-time analysis cache with automatic refresh."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.common.analysis import Analysis, AnalysisOutput

DEFAULT_REFRESH_INTERVAL = 300  # 5 minutes


@dataclass
class CacheEntry:
    """A cached analysis result."""

    key: str
    data: Any
    output: Optional[AnalysisOutput]
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        result = {
            "key": self.key,
            "created_at": datetime.fromtimestamp(self.created_at).isoformat(),
            "data": self.data,
        }
        if self.output and self.output.data is not None:
            result["data"] = self.output.data.to_dict("records")
        return result


class AnalysisCache:
    """Cache for analysis results with automatic refresh."""

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        refresh_interval: int = DEFAULT_REFRESH_INTERVAL,
    ):
        self.base_dir = base_dir or Path.cwd()
        self.trades_dir = self.base_dir / "data" / "kalshi" / "trades"
        self.markets_dir = self.base_dir / "data" / "kalshi" / "markets"
        self.analytics_dir = self.base_dir / "data" / "analytics"
        self.analytics_dir.mkdir(parents=True, exist_ok=True)
        self.refresh_interval = refresh_interval
        self._cache: dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        # Best-effort load of any persisted analysis snapshots so the API
        # can serve warm data immediately after process start.
        self._load_persisted_snapshots()

    def _generate_key(self, analysis_name: str, params: Optional[dict] = None) -> str:
        """Generate a cache key for an analysis."""
        key_data = {
            "analysis": analysis_name,
            "trades_dir": str(self.trades_dir),
            "markets_dir": str(self.markets_dir),
            "params": params or {},
        }
        json_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    async def get_or_refresh(
        self,
        analysis_name: str,
        params: Optional[dict] = None,
        refresh_interval: Optional[int] = None,
    ) -> CacheEntry:
        """Get cached result or compute and cache it."""
        key = self._generate_key(analysis_name, params)

        # Fast path: return a valid cached entry without doing heavy work.
        async with self._lock:
            entry = self._cache.get(key)
            if entry and not entry.is_expired():
                return entry

        # Compute fresh result outside the lock so multiple analyses can be
        # computed in parallel without blocking each other.
        output = await self._compute_analysis(analysis_name, params)
        expires_at = time.time() + (refresh_interval or self.refresh_interval)

        new_entry = CacheEntry(
            key=key,
            data=self._serialize_output(output),
            output=output,
            expires_at=expires_at,
        )

        # Persist snapshot to disk and store the new entry under the lock.
        self._persist_snapshot(analysis_name, new_entry)
        async with self._lock:
            self._cache[key] = new_entry
            return new_entry

    async def _compute_analysis(self, analysis_name: str, params: Optional[dict]) -> AnalysisOutput:
        """Compute analysis result."""
        # Import analysis classes dynamically
        try:
            module = __import__(f"src.analysis.kalshi.{analysis_name}", fromlist=[""])
        except ImportError:
            raise ValueError(f"Unknown analysis: {analysis_name}")

        # Get the analysis class
        analysis_class = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, Analysis) and attr is not Analysis:
                analysis_class = attr
                break

        if analysis_class is None:
            raise ValueError(f"Could not find analysis class in {analysis_name}")

        # Check what parameters the analysis class accepts
        import inspect
        sig = inspect.signature(analysis_class.__init__)
        init_params = list(sig.parameters.keys())

        # Build kwargs based on accepted parameters
        kwargs = {}
        if "trades_dir" in init_params:
            kwargs["trades_dir"] = self.trades_dir
        if "markets_dir" in init_params:
            kwargs["markets_dir"] = self.markets_dir

        # Instantiate and run analysis. Run the heavy work in a thread so we
        # don't block the event loop that called get_or_refresh.
        analysis = analysis_class(**kwargs)
        return await asyncio.to_thread(analysis.run)

    async def get_cached(
        self,
        analysis_name: str,
        params: Optional[dict] = None,
    ) -> Optional[CacheEntry]:
        """Return the latest cached result if present, without recomputing.

        This intentionally ignores TTL/expiration so that APIs always have a
        "hot" snapshot to serve, even while a fresher version is being
        recomputed in the background. The refresh interval only controls when
        background workers should recompute, not whether an existing value can
        be served.
        """
        key = self._generate_key(analysis_name, params)
        async with self._lock:
            return self._cache.get(key)

    def _serialize_output(self, output: AnalysisOutput) -> dict:
        """Convert AnalysisOutput to serializable dict."""
        result = {"metadata": {}}

        if output.data is not None:
            result["data"] = output.data.to_dict("records")

        if output.chart:
            result["chart"] = output.chart.to_dict()

        if output.figure:
            result["metadata"]["has_figure"] = True

        return result

    def _snapshot_path(self, analysis_name: str) -> Path:
        """Path for persisted snapshot JSON for a given analysis."""
        return self.analytics_dir / f"{analysis_name}.json"

    def _persist_snapshot(self, analysis_name: str, entry: CacheEntry) -> None:
        """Persist a lightweight snapshot of the analysis to disk."""
        try:
            snapshot = {
                "analysis": analysis_name,
                "created_at": datetime.fromtimestamp(entry.created_at).isoformat(),
                "data": entry.data,
            }
            path = self._snapshot_path(analysis_name)
            path.write_text(json.dumps(snapshot, default=str))
        except Exception:
            # Snapshot persistence is best-effort; avoid breaking requests.
            return

    def _load_persisted_snapshots(self) -> None:
        """Load any persisted analysis snapshots into the in-memory cache."""
        try:
            for path in self.analytics_dir.glob("*.json"):
                try:
                    snapshot = json.loads(path.read_text())
                except Exception:
                    continue

                analysis_name = snapshot.get("analysis") or path.stem
                key = self._generate_key(analysis_name, None)

                created_iso = snapshot.get("created_at")
                try:
                    created_ts = (
                        datetime.fromisoformat(created_iso).timestamp()
                        if created_iso
                        else time.time()
                    )
                except Exception:
                    created_ts = time.time()

                entry = CacheEntry(
                    key=key,
                    data=snapshot.get("data", {}),
                    output=None,
                    created_at=created_ts,
                    expires_at=time.time() + self.refresh_interval,
                )
                self._cache[key] = entry
        except Exception:
            # If anything goes wrong, just start with an empty cache.
            return

    async def refresh_all(self) -> dict[str, CacheEntry]:
        """Mark all cached entries as stale without dropping the hot state.

        This forces background workers to recompute analyses soon (because
        their TTL has effectively expired) while allowing APIs to continue
        serving the last known-good snapshot from memory.
        """
        async with self._lock:
            now = time.time()
            for entry in self._cache.values():
                entry.expires_at = now
            return dict(self._cache)

    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "refresh_interval_seconds": self.refresh_interval,
        }


# Global cache instance
_cache: Optional[AnalysisCache] = None


def get_cache(base_dir: Optional[Path] = None) -> AnalysisCache:
    """Get or create the global cache instance."""
    global _cache
    if _cache is None:
        _cache = AnalysisCache(base_dir=base_dir)
    return _cache