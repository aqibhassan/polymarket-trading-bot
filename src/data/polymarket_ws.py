"""Polymarket CLOB WebSocket feed."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
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
    # Per-field timestamps for granular freshness checks
    bid_ask_updated: datetime | None = None
    last_trade_updated: datetime | None = None
    midpoint_updated: datetime | None = None


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
        self._liquidity_callbacks: list[Callable] = []
        self._backoff = 1.0
        self._max_backoff = 60.0

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_clob_state(self, token_id: str) -> CLOBState | None:
        """Get current CLOB state for a token. Returns None if not tracked."""
        return self._clob_state.get(token_id)

    def register_liquidity_callback(self, callback: Callable) -> None:
        """Register callback for when real liquidity appears on the book.

        Callback signature: (token_id: str, best_bid: Decimal, best_ask: Decimal, size: Decimal) -> None
        Async callbacks are supported — they will be scheduled via asyncio.create_task().
        """
        self._liquidity_callbacks.append(callback)

    def seed_rest_data(
        self,
        token_id: str,
        last_trade_price: Any,
        midpoint: Any,
        spread: Any,
    ) -> None:
        """Populate CLOBState from REST API when WS hasn't received data yet.

        Only seeds fields that are currently None (WS data takes priority).
        """
        state = self._clob_state.setdefault(token_id, CLOBState())
        now = datetime.now(tz=UTC)

        # Parse last_trade_price
        if state.last_trade_price is None and last_trade_price is not None:
            try:
                _val = last_trade_price
                if isinstance(_val, dict):
                    _val = _val.get("price", _val)
                _parsed = Decimal(str(_val))
                if _parsed > 0:
                    state.last_trade_price = _parsed
                    state.last_trade_updated = now
                    log.info("seed_rest_last_trade", token=token_id[:16], price=str(_parsed))
            except (ValueError, TypeError, ArithmeticError):
                pass

        # Parse midpoint
        if state.midpoint is None and midpoint is not None:
            try:
                _val = midpoint
                if isinstance(_val, dict):
                    _val = _val.get("mid", _val)
                _parsed = Decimal(str(_val))
                if _parsed > 0:
                    state.midpoint = _parsed
                    state.midpoint_updated = now
                    log.info("seed_rest_midpoint", token=token_id[:16], mid=str(_parsed))
            except (ValueError, TypeError, ArithmeticError):
                pass

        # Parse spread → derive best_bid/best_ask from midpoint if possible
        if state.best_bid is None and state.midpoint is not None and spread is not None:
            try:
                _val = spread
                if isinstance(_val, dict):
                    _val = _val.get("spread", _val)
                _sp = Decimal(str(_val))
                if _sp >= 0:
                    state.best_bid = state.midpoint - _sp / 2
                    state.best_ask = state.midpoint + _sp / 2
                    state.bid_ask_updated = now
            except (ValueError, TypeError, ArithmeticError):
                pass

        if state.last_updated is None:
            state.last_updated = now

    def update_rest_data(
        self,
        token_id: str,
        last_trade_price: Any,
        midpoint: Any,
        spread: Any,
    ) -> None:
        """Refresh CLOBState from REST API — always updates timestamps.

        Unlike seed_rest_data (which only sets None fields), this method
        overwrites existing values. Used by periodic REST re-polling to
        keep data fresh on illiquid markets where WS delivers no events.
        """
        state = self._clob_state.setdefault(token_id, CLOBState())
        now = datetime.now(tz=UTC)

        # Parse last_trade_price — always overwrite
        if last_trade_price is not None:
            try:
                _val = last_trade_price
                if isinstance(_val, dict):
                    _val = _val.get("price", _val)
                _parsed = Decimal(str(_val))
                if _parsed > 0:
                    state.last_trade_price = _parsed
                    state.last_trade_updated = now
                    log.info("update_rest_last_trade", token=token_id[:16], price=str(_parsed))
            except (ValueError, TypeError, ArithmeticError):
                pass

        # Parse midpoint — always overwrite
        if midpoint is not None:
            try:
                _val = midpoint
                if isinstance(_val, dict):
                    _val = _val.get("mid", _val)
                _parsed = Decimal(str(_val))
                if _parsed > 0:
                    state.midpoint = _parsed
                    state.midpoint_updated = now
                    log.info("update_rest_midpoint", token=token_id[:16], mid=str(_parsed))
            except (ValueError, TypeError, ArithmeticError):
                pass

        # Parse spread → derive best_bid/best_ask — always overwrite
        if state.midpoint is not None and spread is not None:
            try:
                _val = spread
                if isinstance(_val, dict):
                    _val = _val.get("spread", _val)
                _sp = Decimal(str(_val))
                if _sp >= 0:
                    state.best_bid = state.midpoint - _sp / 2
                    state.best_ask = state.midpoint + _sp / 2
                    state.bid_ask_updated = now
            except (ValueError, TypeError, ArithmeticError):
                pass

        state.last_updated = now

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
        state.bid_ask_updated = now
        state.midpoint = midpoint
        if midpoint is not None:
            state.midpoint_updated = now
        state.last_updated = now

        # Also emit orderbook if callback is set
        if self._on_orderbook is not None:
            self._emit_orderbook(data, now)

    def _handle_price_change(self, data: dict[str, Any], now: datetime) -> None:
        """Handle incremental price change event (includes best_bid/best_ask).

        Supports two Polymarket formats:

        1. **Nested ``price_changes``** (primary format)::

            {"event_type": "price_change", "market": "cond_id",
             "price_changes": [{"asset_id": "...", "price": "0.55",
              "size": "50", "side": "BUY", "best_bid": "0.55", "best_ask": "0.60"}]}

        2. **Flat format** — top-level ``asset_id``, ``price``, ``side``,
           ``best_bid``, ``best_ask``, and/or ``changes`` array of
           ``[side, price, size]`` tuples.

        When real liquidity is detected (non-placeholder, spread < 0.80),
        invokes all registered liquidity callbacks.
        """
        condition_id = data.get("market", "")

        # --- Handle nested price_changes array (primary Polymarket format) ---
        price_changes = data.get("price_changes", [])
        if price_changes:
            for pc in price_changes:
                if not isinstance(pc, dict):
                    continue
                pc_asset = pc.get("asset_id", "")
                if not pc_asset:
                    continue
                self._apply_price_change_entry(pc_asset, pc, now, condition_id)
            return

        # --- Flat format: top-level asset_id ---
        asset_id = data.get("asset_id", "")
        if not asset_id:
            return
        self._apply_price_change_entry(asset_id, data, now, condition_id)

    def _apply_price_change_entry(
        self,
        asset_id: str,
        entry: dict[str, Any],
        now: datetime,
        condition_id: str = "",
    ) -> None:
        """Apply a single price-change entry (from nested or flat format)."""
        state = self._clob_state.setdefault(asset_id, CLOBState())
        max_change_size = Decimal("0")

        # Top-level price/side
        price_str = entry.get("price")
        if price_str is not None:
            try:
                price = Decimal(str(price_str))
                side = entry.get("side", "").lower()
                if side in ("buy", "bid"):
                    state.best_bid = price
                elif side in ("sell", "ask"):
                    state.best_ask = price
            except (ValueError, TypeError, ArithmeticError):
                pass

        # Size tracking
        size_str = entry.get("size")
        if size_str is not None:
            try:
                max_change_size = Decimal(str(size_str))
            except (ValueError, TypeError, ArithmeticError):
                pass

        # Parse changes array: each element is [side, price, size]
        changes = entry.get("changes", [])
        for change in changes:
            if not isinstance(change, list) or len(change) < 3:
                continue
            c_side, c_price_str, c_size_str = change[0], change[1], change[2]
            try:
                c_price = Decimal(str(c_price_str))
                c_size = Decimal(str(c_size_str))
            except (ValueError, TypeError, ArithmeticError):
                continue
            c_side_lower = str(c_side).lower()
            if c_side_lower in ("buy", "bid"):
                if state.best_bid is None or c_price > state.best_bid:
                    state.best_bid = c_price
            elif c_side_lower in ("sell", "ask"):
                if state.best_ask is None or c_price < state.best_ask:
                    state.best_ask = c_price
            if c_size > max_change_size:
                max_change_size = c_size

        # Explicit best_bid / best_ask override everything
        if "best_bid" in entry:
            try:
                state.best_bid = Decimal(str(entry["best_bid"]))
            except (ValueError, TypeError, ArithmeticError):
                pass
        if "best_ask" in entry:
            try:
                state.best_ask = Decimal(str(entry["best_ask"]))
            except (ValueError, TypeError, ArithmeticError):
                pass

        if state.best_bid is not None and state.best_ask is not None:
            state.midpoint = (state.best_bid + state.best_ask) / 2
            state.midpoint_updated = now

        state.bid_ask_updated = now
        state.last_updated = now

        # Liquidity detection — invoke callbacks for real (non-placeholder) prices
        if state.best_bid is not None and state.best_ask is not None:
            self._check_liquidity(asset_id, state.best_bid, state.best_ask, max_change_size)

    def _handle_last_trade_price(self, data: dict[str, Any], now: datetime) -> None:
        """Handle trade execution event."""
        asset_id = data.get("asset_id", "")
        if not asset_id:
            return

        state = self._clob_state.setdefault(asset_id, CLOBState())
        price_str = data.get("price")
        if price_str is not None:
            state.last_trade_price = Decimal(str(price_str))
            state.last_trade_updated = now
        state.last_updated = now

    # -- Placeholder thresholds for liquidity detection --
    _PLACEHOLDER_BIDS = frozenset([Decimal("0.01"), Decimal("0.001")])
    _PLACEHOLDER_ASKS = frozenset([Decimal("0.99"), Decimal("0.999")])
    _MAX_REAL_SPREAD = Decimal("0.80")
    _MIN_REAL_BID = Decimal("0.10")
    _MAX_REAL_ASK = Decimal("0.90")

    def _check_liquidity(
        self,
        token_id: str,
        best_bid: Decimal,
        best_ask: Decimal,
        size: Decimal,
    ) -> None:
        """Check whether prices represent real liquidity and invoke callbacks.

        Filters out placeholder prices (0.01/0.99, 0.001/0.999) and only
        triggers when spread < 0.80, bid > 0.10, ask < 0.90.
        """
        if best_bid in self._PLACEHOLDER_BIDS or best_ask in self._PLACEHOLDER_ASKS:
            return
        if best_bid < self._MIN_REAL_BID or best_ask > self._MAX_REAL_ASK:
            return
        spread = best_ask - best_bid
        if spread >= self._MAX_REAL_SPREAD or spread <= 0:
            return

        log.info(
            "ws_price_change_detected",
            token_id=token_id[:16],
            best_bid=str(best_bid),
            best_ask=str(best_ask),
            spread=str(spread),
            size=str(size),
        )

        for cb in self._liquidity_callbacks:
            try:
                result = cb(token_id, best_bid, best_ask, size)
                if inspect.isawaitable(result):
                    asyncio.create_task(result)
            except Exception:
                log.warning("liquidity_callback_error", token_id=token_id[:16], exc_info=True)

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
