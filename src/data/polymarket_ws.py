"""Polymarket CLOB WebSocket feed."""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import websockets

from src.core.logging import get_logger
from src.models.market import MarketState, OrderBookLevel, OrderBookSnapshot

log = get_logger(__name__)


@dataclass
class CLOBState:
    """Per-asset CLOB state maintained from WebSocket events."""

    best_bid: Decimal | None = None
    best_ask: Decimal | None = None
    midpoint: Decimal | None = None
    last_trade_price: Decimal | None = None
    last_updated: datetime | None = None


class PolymarketWSFeed:
    """WebSocket subscriber for Polymarket CLOB market data.

    Implements the DataFeed protocol from src.interfaces.

    Subscribes using token IDs (asset IDs) per Polymarket WS API:
      subscribe:   {"assets_ids": [token_id], "type": "market"}
      unsubscribe: {"assets_ids": [token_id], "type": "market"}
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
        self._subscribed_assets: set[str] = set()
        self._clob_state: dict[str, CLOBState] = {}
        self._backoff = 1.0
        self._max_backoff = 60.0

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_clob_state(self, token_id: str) -> CLOBState | None:
        """Get current CLOB state for a token. Returns None if not tracked."""
        return self._clob_state.get(token_id)

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

    async def subscribe(self, token_id: str) -> None:
        """Subscribe to a token's market updates (uses asset/token ID)."""
        self._subscribed_assets.add(token_id)
        if self._connected and self._ws is not None:
            await self._send_subscribe(token_id)

    async def unsubscribe(self, token_id: str) -> None:
        """Unsubscribe from a token's market updates."""
        self._subscribed_assets.discard(token_id)
        if self._connected and self._ws is not None:
            await self._send_unsubscribe(token_id)
        self._clob_state.pop(token_id, None)

    async def _send_subscribe(self, token_id: str) -> None:
        """Send subscription message for a token (Polymarket WS format)."""
        if self._ws is None:
            return
        try:
            msg = json.dumps({
                "assets_ids": [token_id],
                "type": "market",
            })
            await asyncio.wait_for(self._ws.send(msg), timeout=5.0)
            log.info("polymarket_ws.subscribed", token_id=token_id[:16])
        except Exception:
            log.warning("polymarket_ws.subscribe_send_failed", token_id=token_id[:16], exc_info=True)

    async def _send_unsubscribe(self, token_id: str) -> None:
        """Send unsubscribe message for a token."""
        if self._ws is None:
            return
        try:
            msg = json.dumps({
                "assets_ids": [token_id],
                "type": "market",
                "action": "unsubscribe",
            })
            await asyncio.wait_for(self._ws.send(msg), timeout=5.0)
            log.info("polymarket_ws.unsubscribed", token_id=token_id[:16])
        except Exception:
            log.warning("polymarket_ws.unsubscribe_send_failed", token_id=token_id[:16], exc_info=True)

    async def _connection_loop(self) -> None:
        """Main loop: connect, receive, reconnect on failure."""
        while self._running:
            try:
                async with websockets.connect(
                    self._ws_url,
                    ping_interval=10,
                    ping_timeout=10,
                ) as ws:
                    self._ws = ws
                    self._connected = True
                    self._backoff = 1.0
                    log.info("polymarket_ws.connected", url=self._ws_url)

                    # Re-subscribe to all assets
                    for token_id in self._subscribed_assets:
                        await self._send_subscribe(token_id)

                    async for raw_msg in ws:
                        if not self._running:
                            break
                        try:
                            self._handle_message(raw_msg)
                        except Exception:
                            log.error("polymarket_ws.handle_error", raw=str(raw_msg)[:200], exc_info=True)

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
            data: list[dict[str, Any]] | dict[str, Any] = json.loads(raw_msg)
        except (json.JSONDecodeError, TypeError):
            log.warning("polymarket_ws.invalid_json", raw=str(raw_msg)[:200])
            return

        # Polymarket sends arrays of events
        if isinstance(data, list):
            for event in data:
                self._handle_event(event)
        elif isinstance(data, dict):
            self._handle_event(data)

    def _handle_event(self, data: dict[str, Any]) -> None:
        """Route a single event by its event_type."""
        event_type = data.get("event_type", "")
        now = datetime.now(tz=UTC)

        if event_type == "book":
            self._handle_book(data, now)
        elif event_type == "price_change":
            self._handle_price_change(data, now)
        elif event_type == "last_trade_price":
            self._handle_last_trade_price(data, now)
        elif event_type == "tick_size_change":
            # Informational — ignore
            pass
        else:
            # Legacy compat: check "type" field for old-style messages
            msg_type = data.get("type", "")
            if msg_type == "market" and self._on_market_state is not None:
                self._emit_market_state(data, now)
            elif msg_type == "book" and self._on_orderbook is not None:
                self._emit_orderbook(data, now)

    def _handle_book(self, data: dict[str, Any], now: datetime) -> None:
        """Handle full orderbook snapshot event."""
        asset_id = data.get("asset_id", "")
        if not asset_id:
            return

        bids = data.get("bids", [])
        asks = data.get("asks", [])

        best_bid: Decimal | None = None
        best_ask: Decimal | None = None

        if bids:
            best_bid = Decimal(str(bids[0].get("price", "0")))
        if asks:
            best_ask = Decimal(str(asks[0].get("price", "0")))

        midpoint: Decimal | None = None
        if best_bid is not None and best_ask is not None:
            midpoint = (best_bid + best_ask) / 2

        state = self._clob_state.setdefault(asset_id, CLOBState())
        state.best_bid = best_bid
        state.best_ask = best_ask
        state.midpoint = midpoint
        state.last_updated = now

        # Also emit orderbook if callback is set
        if self._on_orderbook is not None:
            self._emit_orderbook(data, now)

    def _handle_price_change(self, data: dict[str, Any], now: datetime) -> None:
        """Handle incremental price change event (includes best_bid/best_ask)."""
        asset_id = data.get("asset_id", "")
        if not asset_id:
            return

        state = self._clob_state.setdefault(asset_id, CLOBState())

        changes = data.get("changes", [])
        # changes is a list of [side, price, size] arrays
        # We update best_bid/best_ask from the "price" field in the event
        price_str = data.get("price")
        if price_str is not None:
            price = Decimal(str(price_str))
            side = data.get("side", "").lower()
            if side == "buy" or side == "bid":
                state.best_bid = price
            elif side == "sell" or side == "ask":
                state.best_ask = price

        # Some events carry explicit best_bid / best_ask
        if "best_bid" in data:
            state.best_bid = Decimal(str(data["best_bid"]))
        if "best_ask" in data:
            state.best_ask = Decimal(str(data["best_ask"]))

        if state.best_bid is not None and state.best_ask is not None:
            state.midpoint = (state.best_bid + state.best_ask) / 2

        state.last_updated = now

    def _handle_last_trade_price(self, data: dict[str, Any], now: datetime) -> None:
        """Handle trade execution event."""
        asset_id = data.get("asset_id", "")
        if not asset_id:
            return

        state = self._clob_state.setdefault(asset_id, CLOBState())
        price_str = data.get("price")
        if price_str is not None:
            state.last_trade_price = Decimal(str(price_str))
        state.last_updated = now

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
                market_id=data.get("market_id", data.get("asset_id", "")),
            )
            self._on_orderbook(snapshot)
        except Exception:
            log.warning("polymarket_ws.book_parse_error", exc_info=True)
