"""Token bucket rate limiter for exchange API calls."""

from __future__ import annotations

import asyncio
import time

from src.core.logging import get_logger

logger = get_logger(__name__)

_SAFETY_MARGIN = 0.8


class TokenBucketRateLimiter:
    """Token bucket rate limiter with 20% safety margin.

    The effective limit is configured_limit * 0.8 to stay safely
    below exchange rate limits.

    All public methods are async-safe via an internal asyncio.Lock
    to prevent race conditions when multiple coroutines call acquire()
    concurrently.
    """

    def __init__(self, max_tokens: int, refill_rate: float) -> None:
        self._max_tokens = int(max_tokens * _SAFETY_MARGIN)
        self._refill_rate = refill_rate * _SAFETY_MARGIN
        self._tokens = float(self._max_tokens)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    @property
    def tokens_remaining(self) -> int:
        self._refill()
        return int(self._tokens)

    async def acquire(self) -> bool:
        """Try to acquire a token. Returns True if successful."""
        async with self._lock:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False

    async def wait(self) -> None:
        """Wait until a token is available, then acquire it."""
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait_time = (
                    (1.0 - self._tokens) / self._refill_rate
                    if self._refill_rate > 0
                    else 0.1
                )
            await asyncio.sleep(min(wait_time, 1.0))

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(
            float(self._max_tokens),
            self._tokens + elapsed * self._refill_rate,
        )
        self._last_refill = now
