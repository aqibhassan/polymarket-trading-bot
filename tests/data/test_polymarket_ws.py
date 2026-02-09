"""Tests for PolymarketWSFeed."""

from __future__ import annotations

import json
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from src.data.polymarket_ws import PolymarketWSFeed
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


class TestPolymarketWSFeedMessageHandling:
    def test_market_state_message(self) -> None:
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

    def test_orderbook_message(self) -> None:
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

    def test_unknown_message_type_ignored(self) -> None:
        state_received: list[MarketState] = []
        book_received: list[OrderBookSnapshot] = []
        feed = PolymarketWSFeed(
            on_market_state=lambda s: state_received.append(s),
            on_orderbook=lambda b: book_received.append(b),
        )

        feed._handle_message(json.dumps({"type": "heartbeat"}))
        assert len(state_received) == 0
        assert len(book_received) == 0

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

    def test_malformed_market_data_handled(self) -> None:
        received: list[MarketState] = []
        feed = PolymarketWSFeed(on_market_state=lambda s: received.append(s))

        # Missing required fields — should log warning, not crash
        msg = json.dumps({
            "type": "market",
            "market_id": "mkt-123",
            # missing yes_price, no_price, time_remaining_seconds
        })
        feed._handle_message(msg)
        # Depending on defaults, this may or may not produce a valid MarketState
        # The important thing is no unhandled exception


class TestPolymarketWSFeedSubscribe:
    @pytest.mark.asyncio()
    async def test_subscribe_adds_market(self) -> None:
        feed = PolymarketWSFeed()
        await feed.subscribe("mkt-123")
        assert "mkt-123" in feed._subscribed_markets

    @pytest.mark.asyncio()
    async def test_disconnect_when_not_connected(self) -> None:
        feed = PolymarketWSFeed()
        await feed.disconnect()
        assert feed.is_connected is False
