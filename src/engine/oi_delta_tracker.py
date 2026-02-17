"""Open Interest Delta Tracker for Binance BTCUSDT futures.

Polls Binance /fapi/v1/openInterest at regular intervals and tracks
OI changes to confirm directional moves with new money flow.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque

import httpx

from src.core.logging import get_logger

logger = get_logger(__name__)


class OIDeltaTracker:
    """Track open interest changes on Binance BTCUSDT futures."""

    def __init__(self, poll_interval: float = 30.0) -> None:
        self._poll_interval = poll_interval
        self._history: deque[tuple[float, float]] = deque(maxlen=60)
        self._running = False
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start background polling."""
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        """Stop background polling."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _poll_loop(self) -> None:
        async with httpx.AsyncClient(timeout=10.0) as client:
            while self._running:
                try:
                    resp = await client.get(
                        "https://fapi.binance.com/fapi/v1/openInterest",
                        params={"symbol": "BTCUSDT"},
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        oi = float(data["openInterest"])
                        self._history.append((time.time(), oi))
                    else:
                        logger.debug("oi_delta_poll_error", status=resp.status_code)
                except asyncio.CancelledError:
                    break
                except Exception:
                    logger.debug("oi_delta_poll_exception", exc_info=True)
                await asyncio.sleep(self._poll_interval)

    def get_delta(self, window_seconds: float = 900.0) -> float:
        """OI change over window (positive = rising OI)."""
        if len(self._history) < 2:
            return 0.0
        now = time.time()
        cutoff = now - window_seconds
        window_samples = [(t, oi) for t, oi in self._history if t >= cutoff]
        if len(window_samples) < 2:
            return 0.0
        first_oi = window_samples[0][1]
        if first_oi == 0:
            return 0.0
        return (window_samples[-1][1] - first_oi) / first_oi

    def get_direction_with_price(self, price_direction: str) -> str:
        """Combine OI delta with price direction for conviction signal."""
        delta = self.get_delta()
        if abs(delta) < 0.001:
            return "neutral"
        if delta > 0:
            return price_direction
        else:
            return "neutral"
