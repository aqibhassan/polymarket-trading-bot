"""Polymarket market resolver — finds active BTC 15-min candle markets."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import httpx

from src.core.logging import get_logger

log = get_logger(__name__)

GAMMA_API_URL = "https://gamma-api.polymarket.com"


@dataclass(frozen=True)
class ResolvedMarket:
    """A resolved Polymarket market for BTC candle trading."""

    market_id: str
    condition_id: str
    yes_token_id: str
    no_token_id: str
    question: str


class MarketResolver:
    """Find active BTC 15-minute candle markets on Polymarket.

    Uses the Gamma API to search for matching markets.
    Results are cached with a configurable TTL.
    """

    def __init__(
        self,
        gamma_url: str = GAMMA_API_URL,
        cache_ttl: int = 120,
        rate_limit_rps: float = 2.0,
    ) -> None:
        self._gamma_url = gamma_url.rstrip("/")
        self._cache_ttl = cache_ttl
        self._min_interval = 1.0 / rate_limit_rps
        self._last_request_time = 0.0
        self._cache: list[ResolvedMarket] = []
        self._cache_ts: float = 0.0

    async def _rate_limit(self) -> None:
        now = asyncio.get_event_loop().time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            await asyncio.sleep(self._min_interval - elapsed)
        self._last_request_time = asyncio.get_event_loop().time()

    async def find_active_markets(self) -> list[ResolvedMarket]:
        """Search for active BTC 15-minute candle markets.

        Returns cached results if within TTL.
        """
        now = time.monotonic()
        if self._cache and (now - self._cache_ts) < self._cache_ttl:
            log.debug("market_resolver.cache_hit", count=len(self._cache))
            return list(self._cache)

        await self._rate_limit()

        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            resp = await client.get(
                f"{self._gamma_url}/markets",
                params={
                    "closed": "false",
                    "limit": 50,
                },
            )
            resp.raise_for_status()
            markets: list[dict[str, Any]] = resp.json()

        results: list[ResolvedMarket] = []
        for m in markets:
            question = m.get("question", "")
            q_lower = question.lower()
            if "btc" not in q_lower and "bitcoin" not in q_lower:
                continue
            if "15" not in q_lower or "minute" not in q_lower:
                continue

            tokens = m.get("tokens", [])
            yes_id = ""
            no_id = ""
            for t in tokens:
                outcome = t.get("outcome", "").upper()
                if outcome == "YES":
                    yes_id = t.get("token_id", "")
                elif outcome == "NO":
                    no_id = t.get("token_id", "")

            if not yes_id or not no_id:
                continue

            results.append(
                ResolvedMarket(
                    market_id=m.get("id", ""),
                    condition_id=m.get("condition_id", ""),
                    yes_token_id=yes_id,
                    no_token_id=no_id,
                    question=question,
                )
            )

        self._cache = results
        self._cache_ts = now
        log.info("market_resolver.resolved", count=len(results))
        return list(results)
