"""Tests for PolymarketWSFeed."""

from __future__ import annotations

import json
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from src.data.polymarket_ws import CLOBState, PolymarketWSFeed
from src.models.market import MarketState, OrderBookSnapshot  # noqa: TCH001


class TestPolymarketWSFeedInit:
    def test_default_params(self) -> None:
        feed = PolymarketWSFeed()
        assert feed.is_connected is False
        assert feed._ws_url == "wss://ws-subscriptions-clob.polymarket.com/ws/market"

    def test_custom_callbacks(self) -> None:
        on_state = MagicMock()
        on_book = MagicMock()
        feed = PolymarketWSFeed(on_market_state=on_state, on_orderbook=on_book)
        assert feed._on_market_state is on_state
        assert feed._on_orderbook is on_book


class TestCLOBState:
    def test_default_values(self) -> None:
        state = CLOBState()
        assert state.best_bid is None
        assert state.best_ask is None
        assert state.midpoint is None
        assert state.last_trade_price is None
        assert state.last_updated is None

    def test_get_clob_state_empty(self) -> None:
        feed = PolymarketWSFeed()
        assert feed.get_clob_state("nonexistent") is None


class TestPolymarketWSSubscription:
    @pytest.mark.asyncio()
    async def test_subscribe_adds_asset(self) -> None:
        feed = PolymarketWSFeed()
        await feed.subscribe("token-abc-123")
        assert "token-abc-123" in feed._subscribed_assets

    @pytest.mark.asyncio()
    async def test_subscribe_format(self) -> None:
        """Subscription message uses correct Polymarket WS format."""
        from unittest.mock import AsyncMock

        feed = PolymarketWSFeed()
        feed._connected = True
        feed._ws = MagicMock()
        feed._ws.send = AsyncMock()

        await feed.subscribe("token-abc-123")
        call_args = feed._ws.send.call_args
        sent = json.loads(call_args[0][0])
        assert sent == {"assets_ids": ["token-abc-123"], "type": "market"}

    @pytest.mark.asyncio()
    async def test_unsubscribe_removes_asset(self) -> None:
        feed = PolymarketWSFeed()
        feed._subscribed_assets.add("token-abc-123")
        await feed.unsubscribe("token-abc-123")
        assert "token-abc-123" not in feed._subscribed_assets

    @pytest.mark.asyncio()
    async def test_unsubscribe_clears_clob_state(self) -> None:
        feed = PolymarketWSFeed()
        feed._clob_state["token-abc-123"] = CLOBState(best_bid=Decimal("0.60"))
        feed._subscribed_assets.add("token-abc-123")
        await feed.unsubscribe("token-abc-123")
        assert "token-abc-123" not in feed._clob_state

    @pytest.mark.asyncio()
    async def test_disconnect_when_not_connected(self) -> None:
        feed = PolymarketWSFeed()
        await feed.disconnect()
        assert feed.is_connected is False


class TestPolymarketWSBookEvent:
    def test_book_event_updates_clob_state(self) -> None:
        feed = PolymarketWSFeed()
        msg = json.dumps([{
            "event_type": "book",
            "asset_id": "token-123",
            "bids": [
                {"price": "0.60", "size": "500"},
                {"price": "0.59", "size": "200"},
            ],
            "asks": [
                {"price": "0.62", "size": "300"},
                {"price": "0.63", "size": "100"},
            ],
        }])
        feed._handle_message(msg)

        state = feed.get_clob_state("token-123")
        assert state is not None
        assert state.best_bid == Decimal("0.60")
        assert state.best_ask == Decimal("0.62")
        assert state.midpoint == Decimal("0.61")
        assert state.last_updated is not None

    def test_book_event_only_bids(self) -> None:
        feed = PolymarketWSFeed()
        msg = json.dumps([{
            "event_type": "book",
            "asset_id": "token-123",
            "bids": [{"price": "0.55", "size": "100"}],
            "asks": [],
        }])
        feed._handle_message(msg)

        state = feed.get_clob_state("token-123")
        assert state is not None
        assert state.best_bid == Decimal("0.55")
        assert state.best_ask is None
        assert state.midpoint is None


class TestPolymarketWSPriceChange:
    def test_price_change_with_explicit_best_bid_ask(self) -> None:
        feed = PolymarketWSFeed()
        msg = json.dumps([{
            "event_type": "price_change",
            "asset_id": "token-123",
            "best_bid": "0.61",
            "best_ask": "0.63",
        }])
        feed._handle_message(msg)

        state = feed.get_clob_state("token-123")
        assert state is not None
        assert state.best_bid == Decimal("0.61")
        assert state.best_ask == Decimal("0.63")
        assert state.midpoint == Decimal("0.62")

    def test_price_change_incremental_update(self) -> None:
        feed = PolymarketWSFeed()
        # Seed initial state
        feed._clob_state["token-123"] = CLOBState(
            best_bid=Decimal("0.60"),
            best_ask=Decimal("0.62"),
        )
        msg = json.dumps([{
            "event_type": "price_change",
            "asset_id": "token-123",
            "best_ask": "0.64",
        }])
        feed._handle_message(msg)

        state = feed.get_clob_state("token-123")
        assert state is not None
        assert state.best_bid == Decimal("0.60")  # unchanged
        assert state.best_ask == Decimal("0.64")  # updated
        assert state.midpoint == Decimal("0.62")  # recalculated


class TestPolymarketWSLastTradePrice:
    def test_last_trade_price_event(self) -> None:
        feed = PolymarketWSFeed()
        msg = json.dumps([{
            "event_type": "last_trade_price",
            "asset_id": "token-123",
            "price": "0.615",
        }])
        feed._handle_message(msg)

        state = feed.get_clob_state("token-123")
        assert state is not None
        assert state.last_trade_price == Decimal("0.615")

    def test_missing_asset_id_ignored(self) -> None:
        feed = PolymarketWSFeed()
        msg = json.dumps([{
            "event_type": "last_trade_price",
            "price": "0.50",
        }])
        feed._handle_message(msg)
        # No crash, no state created
        assert len(feed._clob_state) == 0


class TestPolymarketWSLegacyMessages:
    def test_legacy_market_state_message(self) -> None:
        received: list[MarketState] = []
        feed = PolymarketWSFeed(on_market_state=lambda s: received.append(s))

        msg = json.dumps({
            "type": "market",
            "market_id": "mkt-123",
            "condition_id": "cond-123",
            "yes_token_id": "yes-tok",
            "no_token_id": "no-tok",
            "yes_price": "0.62",
            "no_price": "0.38",
            "time_remaining_seconds": 600,
            "question": "Will BTC go up?",
        })
        feed._handle_message(msg)

        assert len(received) == 1
        state = received[0]
        assert state.market_id == "mkt-123"
        assert state.yes_price == Decimal("0.62")
        assert state.no_price == Decimal("0.38")

    def test_legacy_orderbook_message(self) -> None:
        received: list[OrderBookSnapshot] = []
        feed = PolymarketWSFeed(on_orderbook=lambda b: received.append(b))

        msg = json.dumps({
            "type": "book",
            "market_id": "mkt-123",
            "bids": [
                {"price": "0.55", "size": "100"},
                {"price": "0.54", "size": "200"},
            ],
            "asks": [
                {"price": "0.57", "size": "150"},
            ],
        })
        feed._handle_message(msg)

        assert len(received) == 1
        book = received[0]
        assert len(book.bids) == 2
        assert len(book.asks) == 1
        assert book.bids[0].price == Decimal("0.55")


class TestPolymarketWSEdgeCases:
    def test_unknown_event_type_ignored(self) -> None:
        feed = PolymarketWSFeed(
            on_market_state=MagicMock(),
            on_orderbook=MagicMock(),
        )
        feed._handle_message(json.dumps([{"event_type": "unknown_type"}]))
        # No exception raised

    def test_invalid_json_ignored(self) -> None:
        feed = PolymarketWSFeed(on_market_state=MagicMock())
        feed._handle_message("not valid json {{{")
        # No exception raised

    def test_no_callback_no_error(self) -> None:
        feed = PolymarketWSFeed()  # No callbacks
        msg = json.dumps({
            "type": "market",
            "market_id": "mkt-123",
            "yes_price": "0.5",
            "no_price": "0.5",
            "time_remaining_seconds": 900,
        })
        feed._handle_message(msg)
        # No exception raised

    def test_array_of_events(self) -> None:
        """Polymarket sends arrays of events — all should be processed."""
        feed = PolymarketWSFeed()
        msg = json.dumps([
            {
                "event_type": "book",
                "asset_id": "token-1",
                "bids": [{"price": "0.50", "size": "100"}],
                "asks": [{"price": "0.52", "size": "100"}],
            },
            {
                "event_type": "last_trade_price",
                "asset_id": "token-1",
                "price": "0.51",
            },
        ])
        feed._handle_message(msg)

        state = feed.get_clob_state("token-1")
        assert state is not None
        assert state.best_bid == Decimal("0.50")
        assert state.last_trade_price == Decimal("0.51")

    def test_tick_size_change_ignored(self) -> None:
        feed = PolymarketWSFeed()
        msg = json.dumps([{
            "event_type": "tick_size_change",
            "asset_id": "token-123",
            "tick_size": "0.01",
        }])
        feed._handle_message(msg)
        # No crash, no state mutation
        assert feed.get_clob_state("token-123") is None
