"""Polymarket CLOB REST client."""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import httpx

from src.core.logging import get_logger
from src.models.market import MarketState, OrderBookLevel, OrderBookSnapshot

log = get_logger(__name__)


def _mask_key(key: str) -> str:
    """Mask all but the last 4 characters of a key."""
    if len(key) <= 4:
        return "****"
    return "*" * (len(key) - 4) + key[-4:]


class PolymarketClient:
    """Async REST client for the Polymarket CLOB API.

    Reads credentials from environment variables:
    - POLYMARKET_API_KEY
    - POLYMARKET_SECRET
    - POLYMARKET_PASSPHRASE
    """

    def __init__(
        self,
        clob_url: str = "https://clob.polymarket.com",
        rate_limit_rps: float = 5.0,
    ) -> None:
        self._clob_url = clob_url.rstrip("/")
        self._api_key = os.environ.get("POLYMARKET_API_KEY", "")
        self._secret = os.environ.get("POLYMARKET_SECRET", "")
        self._passphrase = os.environ.get("POLYMARKET_PASSPHRASE", "")
        self._min_interval = 1.0 / rate_limit_rps
        self._last_request_time = 0.0
        self._client: httpx.AsyncClient | None = None

        if self._api_key:
            log.info(
                "polymarket_client.init",
                api_key=_mask_key(self._api_key),
            )

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._clob_url,
                headers=self._auth_headers(),
                timeout=httpx.Timeout(10.0),
            )
        return self._client

    def _auth_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._api_key:
            headers["POLY_API_KEY"] = self._api_key
            headers["POLY_PASSPHRASE"] = self._passphrase
            headers["POLY_SECRET"] = self._secret
        return headers

    async def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        now = asyncio.get_event_loop().time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            await asyncio.sleep(self._min_interval - elapsed)
        self._last_request_time = asyncio.get_event_loop().time()

    async def get_market(self, condition_id: str) -> MarketState:
        """Fetch market state by condition ID."""
        await self._rate_limit()
        client = await self._get_client()
        resp = await client.get(f"/markets/{condition_id}")
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()

        tokens = data.get("tokens", [])
        yes_token_id = ""
        no_token_id = ""
        yes_price = Decimal("0.5")
        no_price = Decimal("0.5")
        for token in tokens:
            outcome = token.get("outcome", "").upper()
            if outcome == "YES":
                yes_token_id = token.get("token_id", "")
                yes_price = Decimal(str(token.get("price", "0.5")))
            elif outcome == "NO":
                no_token_id = token.get("token_id", "")
                no_price = Decimal(str(token.get("price", "0.5")))

        return MarketState(
            market_id=data.get("condition_id", condition_id),
            condition_id=condition_id,
            yes_token_id=yes_token_id,
            no_token_id=no_token_id,
            yes_price=yes_price,
            no_price=no_price,
            time_remaining_seconds=int(data.get("seconds_until_close", 900)),
            question=data.get("question", ""),
        )

    async def get_orderbook(self, token_id: str) -> OrderBookSnapshot:
        """Fetch current order book for a token."""
        await self._rate_limit()
        client = await self._get_client()
        resp = await client.get("/book", params={"token_id": token_id})
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
            timestamp=datetime.now(tz=UTC),
            market_id=token_id,
        )

    async def place_order(
        self,
        token_id: str,
        side: str,
        price: Decimal,
        size: Decimal,
    ) -> dict[str, Any]:
        """Place a limit order on the CLOB."""
        await self._rate_limit()
        client = await self._get_client()
        payload = {
            "tokenID": token_id,
            "side": side,
            "price": str(price),
            "size": str(size),
        }
        log.info(
            "polymarket_client.place_order",
            token_id=token_id,
            side=side,
            price=str(price),
            size=str(size),
        )
        resp = await client.post("/order", json=payload)
        resp.raise_for_status()
        result: dict[str, Any] = resp.json()
        return result

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        await self._rate_limit()
        client = await self._get_client()
        resp = await client.delete(f"/order/{order_id}")
        return resp.status_code == 200

    async def get_balance(self) -> Decimal:
        """Fetch USDC balance."""
        await self._rate_limit()
        client = await self._get_client()
        resp = await client.get("/balance")
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return Decimal(str(data.get("balance", "0")))

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
