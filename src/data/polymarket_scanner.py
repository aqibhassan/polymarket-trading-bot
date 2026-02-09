"""Polymarket market scanner -- finds active BTC 15-min candle prediction markets."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import httpx

from src.core.logging import get_logger
from src.models.market import MarketState, OrderBookLevel, OrderBookSnapshot

log = get_logger(__name__)

GAMMA_API_URL = "https://gamma-api.polymarket.com"

# Outcome mapping: Polymarket BTC 15m markets use "Up"/"Down" not "Yes"/"No"
_OUTCOME_MAP = {"UP": "YES", "DOWN": "NO", "YES": "YES", "NO": "NO"}


class PolymarketScanner:
    """Scan Gamma Markets API for active BTC 15-minute candle prediction markets.

    Features:
    - Filters for BTC 15m candle markets by question text
    - Rate limiting (max 10 req/s)
    - 5-second result caching to avoid redundant calls
    - Minimum volume filtering
    """

    def __init__(
        self,
        min_volume: float = 500,
        session: Any = None,
        gamma_url: str = GAMMA_API_URL,
    ) -> None:
        self._min_volume = min_volume
        self._gamma_url = gamma_url.rstrip("/")
        self._session = session
        self._min_interval = 1.0 / 10.0  # max 10 req/s
        self._last_request_time = 0.0

        # Cache for scan results (5s TTL)
        self._scan_cache: list[MarketState] | None = None
        self._scan_cache_ts: float = 0.0
        self._cache_ttl: float = 5.0

        # Cache for market prices (5s TTL)
        self._price_cache: dict[str, tuple[Decimal, Decimal]] = {}
        self._price_cache_ts: dict[str, float] = {}

    async def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        now = asyncio.get_event_loop().time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            await asyncio.sleep(self._min_interval - elapsed)
        self._last_request_time = asyncio.get_event_loop().time()

    def _is_btc_15m_market(self, question: str) -> bool:
        """Check if a market question matches BTC 15-minute candle criteria.

        Matches both legacy format ("BTC 15-minute candle...") and the
        current Polymarket format ("Bitcoin Up or Down - Feb 7, 6:45PM-7:00PM ET").
        """
        q_lower = question.lower()
        has_btc = "btc" in q_lower or "bitcoin" in q_lower
        if not has_btc:
            return False
        # New format: "Bitcoin Up or Down - <date>, <time>-<time> ET"
        if "up or down" in q_lower:
            return True
        # Legacy format: "BTC 15-minute candle..."
        has_15 = "15" in q_lower
        has_candle_or_minute = "candle" in q_lower or "minute" in q_lower
        return has_15 and has_candle_or_minute

    def _parse_market(self, data: dict[str, Any]) -> MarketState | None:
        """Parse a Gamma API market response into a MarketState.

        Handles both legacy format (tokens array with YES/NO) and the current
        BTC 15m format (clobTokenIds array with Up/Down outcomes).

        Returns None if the market cannot be parsed or lacks required fields.
        """
        question = data.get("question", "") or ""
        if not self._is_btc_15m_market(question):
            return None

        # Skip closed markets
        if data.get("closed", False):
            return None

        # Extract token IDs — try clobTokenIds first (current format), then tokens array
        yes_token_id = ""
        no_token_id = ""

        outcomes_raw = data.get("outcomes", "")
        if isinstance(outcomes_raw, str) and outcomes_raw:
            try:
                outcomes = json.loads(outcomes_raw)
            except (json.JSONDecodeError, ValueError):
                outcomes = []
        elif isinstance(outcomes_raw, list):
            outcomes = outcomes_raw
        else:
            outcomes = []

        clob_token_ids_raw = data.get("clobTokenIds", "")
        if isinstance(clob_token_ids_raw, str) and clob_token_ids_raw:
            try:
                clob_token_ids = json.loads(clob_token_ids_raw)
            except (json.JSONDecodeError, ValueError):
                clob_token_ids = []
        elif isinstance(clob_token_ids_raw, list):
            clob_token_ids = clob_token_ids_raw
        else:
            clob_token_ids = []

        if clob_token_ids and outcomes and len(clob_token_ids) >= 2 and len(outcomes) >= 2:
            # Map outcomes to YES/NO using _OUTCOME_MAP
            for i, outcome in enumerate(outcomes):
                mapped = _OUTCOME_MAP.get(outcome.upper(), "")
                if mapped == "YES":
                    yes_token_id = str(clob_token_ids[i])
                elif mapped == "NO":
                    no_token_id = str(clob_token_ids[i])

        # Fallback: try tokens array (legacy format)
        if not yes_token_id or not no_token_id:
            tokens = data.get("tokens", [])
            for token in tokens:
                outcome = token.get("outcome", "").upper()
                mapped = _OUTCOME_MAP.get(outcome, outcome)
                if mapped == "YES":
                    yes_token_id = token.get("token_id", "")
                elif mapped == "NO":
                    no_token_id = token.get("token_id", "")

        if not yes_token_id or not no_token_id:
            log.debug(
                "polymarket_scanner.no_tokens",
                question=question[:60],
                outcomes=outcomes,
                clob_ids=len(clob_token_ids),
            )
            return None

        # Extract prices from outcomePrices (JSON string array)
        yes_price = Decimal("0.5")
        no_price = Decimal("0.5")
        outcome_prices_raw = data.get("outcomePrices", "")
        if outcome_prices_raw:
            try:
                if isinstance(outcome_prices_raw, str):
                    prices = json.loads(outcome_prices_raw)
                else:
                    prices = outcome_prices_raw
                if len(prices) >= 2 and outcomes:
                    # Map prices to YES/NO by outcome name, not by index
                    for i, outcome in enumerate(outcomes):
                        mapped = _OUTCOME_MAP.get(outcome.upper(), outcome.upper())
                        if mapped == "YES" and i < len(prices):
                            yes_price = Decimal(str(prices[i]))
                        elif mapped == "NO" and i < len(prices):
                            no_price = Decimal(str(prices[i]))
                elif len(prices) >= 2:
                    # No outcomes available — log warning, skip index-based assumption
                    log.warning(
                        "polymarket_scanner.no_outcomes_for_prices",
                        market_id=data.get("conditionId", "")[:20],
                        msg="cannot map prices without outcome names, using defaults",
                    )
            except (json.JSONDecodeError, ValueError, IndexError):
                pass

        # Compute time remaining from endDate
        end_date_str = data.get("endDate", "") or data.get("end_date_iso", "")
        time_remaining_seconds = 900  # default 15 minutes
        if end_date_str:
            try:
                end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                now = datetime.now(tz=timezone.utc)
                remaining = (end_dt - now).total_seconds()
                time_remaining_seconds = max(0, int(remaining))
            except (ValueError, TypeError):
                pass

        # Market ID: prefer conditionId, fall back to id
        market_id = data.get("conditionId", "") or data.get("condition_id", "") or str(data.get("id", ""))
        condition_id = data.get("conditionId", "") or data.get("condition_id", "")

        # Volume check (relax for events-API results which may have lower volume on new windows)
        volume_str = data.get("volume", "0") or data.get("volumeNum", "0") or "0"
        try:
            volume = float(str(volume_str))
        except (ValueError, TypeError):
            volume = 0.0

        if volume < self._min_volume:
            log.debug(
                "polymarket_scanner.low_volume",
                market_id=market_id[:20],
                volume=volume,
                min_volume=self._min_volume,
            )
            return None

        return MarketState(
            market_id=market_id,
            condition_id=condition_id,
            yes_token_id=yes_token_id,
            no_token_id=no_token_id,
            yes_price=yes_price,
            no_price=no_price,
            time_remaining_seconds=time_remaining_seconds,
            question=question,
        )

    @staticmethod
    def _window_slugs() -> list[str]:
        """Generate event slugs for current, next, and previous 15-minute windows."""
        now_ts = int(time.time())
        window_start = (now_ts // 900) * 900
        return [
            f"btc-updown-15m-{window_start - 900}",
            f"btc-updown-15m-{window_start}",
            f"btc-updown-15m-{window_start + 900}",
        ]

    async def scan_active_markets(self) -> list[MarketState]:
        """Find currently active BTC 15-minute candle prediction markets.

        Uses the Gamma events API with computed slugs (btc-updown-15m-{ts})
        to reliably discover markets that don't appear in the generic /markets listing.

        Returns cached results if within 5-second TTL.
        """
        now = time.monotonic()
        if self._scan_cache is not None and (now - self._scan_cache_ts) < self._cache_ttl:
            log.debug("polymarket_scanner.cache_hit", count=len(self._scan_cache))
            return list(self._scan_cache)

        slugs = self._window_slugs()
        results: list[MarketState] = []

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
                for slug in slugs:
                    await self._rate_limit()
                    try:
                        resp = await client.get(
                            f"{self._gamma_url}/events",
                            params={"slug": slug},
                        )
                        resp.raise_for_status()
                        events: list[dict[str, Any]] = resp.json()
                    except httpx.HTTPError as exc:
                        log.warning("polymarket_scanner.slug_error", slug=slug, error=str(exc))
                        continue

                    for event in events:
                        for market_data in event.get("markets", []):
                            parsed = self._parse_market(market_data)
                            if parsed is not None:
                                results.append(parsed)
        except httpx.HTTPError as exc:
            log.error("polymarket_scanner.api_error", error=str(exc))
            if self._scan_cache is not None:
                return list(self._scan_cache)
            return []

        # Deduplicate by market_id
        seen: set[str] = set()
        unique: list[MarketState] = []
        for m in results:
            if m.market_id not in seen:
                seen.add(m.market_id)
                unique.append(m)
        results = unique

        self._scan_cache = results
        self._scan_cache_ts = now
        log.info(
            "polymarket_scanner.scanned",
            found=len(results),
            slugs_checked=len(slugs),
        )
        return list(results)

    async def get_market_prices(self, market_id: str) -> tuple[Decimal, Decimal]:
        """Get (yes_price, no_price) for a specific market.

        Returns cached results if within 5-second TTL.
        """
        now = time.monotonic()
        cached_ts = self._price_cache_ts.get(market_id, 0.0)
        if market_id in self._price_cache and (now - cached_ts) < self._cache_ttl:
            return self._price_cache[market_id]

        await self._rate_limit()

        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            resp = await client.get(
                f"{self._gamma_url}/markets/{market_id}",
            )
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()

        yes_price = Decimal("0.5")
        no_price = Decimal("0.5")

        # Parse outcomes for name-based price mapping
        outcomes_raw = data.get("outcomes", "")
        if isinstance(outcomes_raw, str) and outcomes_raw:
            try:
                outcomes = json.loads(outcomes_raw)
            except (json.JSONDecodeError, ValueError):
                outcomes = []
        elif isinstance(outcomes_raw, list):
            outcomes = outcomes_raw
        else:
            outcomes = []

        outcome_prices_raw = data.get("outcomePrices", "")
        if outcome_prices_raw:
            try:
                if isinstance(outcome_prices_raw, str):
                    prices = json.loads(outcome_prices_raw)
                else:
                    prices = outcome_prices_raw
                if len(prices) >= 2 and outcomes:
                    # Map by outcome name — same logic as _parse_market()
                    for i, outcome in enumerate(outcomes):
                        mapped = _OUTCOME_MAP.get(str(outcome).upper(), "")
                        if mapped == "YES" and i < len(prices):
                            yes_price = Decimal(str(prices[i]))
                        elif mapped == "NO" and i < len(prices):
                            no_price = Decimal(str(prices[i]))
                elif len(prices) >= 2:
                    # No outcomes — fall back to index order with warning
                    log.warning(
                        "get_market_prices.no_outcomes",
                        market_id=market_id,
                    )
                    yes_price = Decimal(str(prices[0]))
                    no_price = Decimal(str(prices[1]))
            except (json.JSONDecodeError, ValueError, IndexError):
                pass

        result = (yes_price, no_price)
        self._price_cache[market_id] = result
        self._price_cache_ts[market_id] = now
        return result

    async def get_market_orderbook(self, market_id: str) -> OrderBookSnapshot:
        """Fetch the order book for a market via the CLOB API."""
        await self._rate_limit()

        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            resp = await client.get(
                "https://clob.polymarket.com/book",
                params={"token_id": market_id},
            )
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()

        bids = [
            OrderBookLevel(price=Decimal(str(b["price"])), size=Decimal(str(b["size"])))
            for b in data.get("bids", [])
        ]
        asks = [
            OrderBookLevel(price=Decimal(str(a["price"])), size=Decimal(str(a["size"])))
            for a in data.get("asks", [])
        ]

        return OrderBookSnapshot(
            bids=bids,
            asks=asks,
            timestamp=datetime.now(tz=timezone.utc),
            market_id=market_id,
        )


async def find_best_btc_market(
    min_volume: float = 500,
) -> MarketState | None:
    """Find the most liquid active BTC 15-minute candle market.

    Returns the market with the highest dominant_price (most decisive pricing),
    or None if no active markets are found.
    """
    scanner = PolymarketScanner(min_volume=min_volume)
    markets = await scanner.scan_active_markets()
    if not markets:
        return None

    # Sort by time remaining (prefer markets with more time left for trading)
    # then by dominant_price as tiebreaker
    return max(markets, key=lambda m: (m.time_remaining_seconds, m.dominant_price))
