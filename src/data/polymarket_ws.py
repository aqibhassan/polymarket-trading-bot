"""Polymarket CLOB WebSocket feed."""

from __future__ import annotations

import asyncio
import contextlib
import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import websockets

from src.core.logging import get_logger
from src.models.market import MarketState, OrderBookLevel, OrderBookSnapshot

log = get_logger(__name__)


class PolymarketWSFeed:
    """WebSocket subscriber for Polymarket CLOB market data.

    Implements the DataFeed protocol from src.interfaces.
    """

    def __init__(
        self,
        ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market",
        on_market_state: Callable[[MarketState], None] | None = None,
        on_orderbook: Callable[[OrderBookSnapshot], None] | None = None,
    ) -> None:
        self._ws_url = ws_url
        self._on_market_state = on_market_state
        self._on_orderbook = on_orderbook
        self._ws: Any = None
        self._connected = False
        self._running = False
        self._recv_task: asyncio.Task[None] | None = None
        self._subscribed_markets: set[str] = set()
        self._backoff = 1.0
        self._max_backoff = 60.0

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        """Connect to Polymarket WebSocket."""
        self._running = True
        self._recv_task = asyncio.create_task(self._connection_loop())
        log.info("polymarket_ws.starting", url=self._ws_url)

    async def disconnect(self) -> None:
        """Disconnect and clean up."""
        self._running = False
        if self._recv_task is not None:
            self._recv_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._recv_task
            self._recv_task = None
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
        self._connected = False
        log.info("polymarket_ws.disconnected")

    async def subscribe(self, market_id: str) -> None:
        """Subscribe to a market's updates."""
        self._subscribed_markets.add(market_id)
        if self._connected and self._ws is not None:
            await self._send_subscribe(market_id)

    async def _send_subscribe(self, market_id: str) -> None:
        """Send subscription message for a market."""
        if self._ws is None:
            return
        msg = json.dumps({
            "type": "subscribe",
            "market": market_id,
            "channel": "market",
        })
        await self._ws.send(msg)
        log.info("polymarket_ws.subscribed", market_id=market_id)

    async def _connection_loop(self) -> None:
        """Main loop: connect, receive, reconnect on failure."""
        while self._running:
            try:
                async with websockets.connect(
                    self._ws_url,
                    ping_interval=30,
                    ping_timeout=10,
                ) as ws:
                    self._ws = ws
                    self._connected = True
                    self._backoff = 1.0
                    log.info("polymarket_ws.connected", url=self._ws_url)

                    # Re-subscribe to all markets
                    for market_id in self._subscribed_markets:
                        await self._send_subscribe(market_id)

                    async for raw_msg in ws:
                        if not self._running:
                            break
                        self._handle_message(raw_msg)

            except asyncio.CancelledError:
                break
            except Exception:
                self._connected = False
                if not self._running:
                    break
                log.warning(
                    "polymarket_ws.reconnecting",
                    backoff_s=self._backoff,
                    exc_info=True,
                )
                await asyncio.sleep(self._backoff)
                self._backoff = min(self._backoff * 2, self._max_backoff)

        self._connected = False

    def _handle_message(self, raw_msg: str | bytes) -> None:
        """Parse and route a WebSocket message."""
        try:
            data: dict[str, Any] = json.loads(raw_msg)
        except (json.JSONDecodeError, TypeError):
            log.warning("polymarket_ws.invalid_json", raw=str(raw_msg)[:200])
            return

        msg_type = data.get("type", "")
        now = datetime.now(tz=timezone.utc)

        if msg_type == "market" and self._on_market_state is not None:
            self._emit_market_state(data, now)
        elif msg_type == "book" and self._on_orderbook is not None:
            self._emit_orderbook(data, now)

    def _emit_market_state(self, data: dict[str, Any], now: datetime) -> None:
        """Build and emit a MarketState from WS data."""
        assert self._on_market_state is not None
        try:
            state = MarketState(
                market_id=data.get("market_id", ""),
                condition_id=data.get("condition_id", ""),
                yes_token_id=data.get("yes_token_id", ""),
                no_token_id=data.get("no_token_id", ""),
                yes_price=Decimal(str(data.get("yes_price", "0.5"))),
                no_price=Decimal(str(data.get("no_price", "0.5"))),
                time_remaining_seconds=int(data.get("time_remaining_seconds", 900)),
                question=data.get("question", ""),
            )
            self._on_market_state(state)
        except Exception:
            log.warning("polymarket_ws.market_parse_error", exc_info=True)

    def _emit_orderbook(self, data: dict[str, Any], now: datetime) -> None:
        """Build and emit an OrderBookSnapshot from WS data."""
        assert self._on_orderbook is not None
        try:
            bids = [
                OrderBookLevel(
                    price=Decimal(str(b["price"])),
                    size=Decimal(str(b["size"])),
                )
                for b in data.get("bids", [])
            ]
            asks = [
                OrderBookLevel(
                    price=Decimal(str(a["price"])),
                    size=Decimal(str(a["size"])),
                )
                for a in data.get("asks", [])
            ]
            snapshot = OrderBookSnapshot(
                bids=bids,
                asks=asks,
                timestamp=now,
                market_id=data.get("market_id", ""),
            )
            self._on_orderbook(snapshot)
        except Exception:
            log.warning("polymarket_ws.book_parse_error", exc_info=True)
