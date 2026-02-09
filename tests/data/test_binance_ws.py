"""Tests for BinanceWSFeed."""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data.binance_ws import BinanceWSFeed


def _make_kline_msg(
    open_: str = "50000",
    high: str = "50500",
    low: str = "49800",
    close: str = "50300",
    volume: str = "123.45",
    ts_ms: int = 1704067200000,
    closed: bool = True,
) -> str:
    return json.dumps({
        "e": "kline",
        "k": {
            "t": ts_ms,
            "o": open_,
            "h": high,
            "l": low,
            "c": close,
            "v": volume,
            "x": closed,
        },
    })


class TestBinanceWSFeedInit:
    def test_default_params(self) -> None:
        feed = BinanceWSFeed()
        assert feed._symbol == "BTCUSDT"
        assert feed._interval == "15m"
        assert feed._max_candles == 50
        assert feed.is_connected is False

    def test_custom_params(self) -> None:
        feed = BinanceWSFeed(
            ws_url="wss://custom.url/ws",
            symbol="ETHUSDT",
            interval="5m",
            max_candles=10,
        )
        assert feed._symbol == "ETHUSDT"
        assert feed._interval == "5m"
        assert feed._max_candles == 10


class TestBinanceWSFeedMessageHandling:
    def test_handle_closed_kline(self) -> None:
        feed = BinanceWSFeed()
        msg = _make_kline_msg(closed=True)
        feed._handle_message(msg)
        assert len(feed._candles) == 1
        candle = feed._candles[0]
        assert candle.exchange == "binance"
        assert candle.symbol == "BTCUSDT"
        assert candle.open == Decimal("50000")
        assert candle.close == Decimal("50300")
        assert candle.volume == Decimal("123.45")

    def test_handle_open_kline_not_stored(self) -> None:
        feed = BinanceWSFeed()
        msg = _make_kline_msg(closed=False)
        feed._handle_message(msg)
        assert len(feed._candles) == 0

    def test_handle_invalid_json(self) -> None:
        feed = BinanceWSFeed()
        feed._handle_message("not json")
        assert len(feed._candles) == 0

    def test_handle_missing_kline_key(self) -> None:
        feed = BinanceWSFeed()
        feed._handle_message(json.dumps({"e": "trade"}))
        assert len(feed._candles) == 0

    def test_deque_max_size(self) -> None:
        feed = BinanceWSFeed(max_candles=3)
        for i in range(5):
            msg = _make_kline_msg(
                close=str(50000 + i),
                ts_ms=1704067200000 + i * 900000,
            )
            feed._handle_message(msg)
        assert len(feed._candles) == 3
        # Oldest should have been evicted
        assert feed._candles[0].close == Decimal("50002")


class TestBinanceWSFeedGetCandles:
    @pytest.mark.asyncio()
    async def test_get_candles_returns_recent(self) -> None:
        feed = BinanceWSFeed()
        for i in range(10):
            msg = _make_kline_msg(
                close=str(50000 + i),
                ts_ms=1704067200000 + i * 900000,
            )
            feed._handle_message(msg)

        candles = await feed.get_candles("BTCUSDT", limit=3)
        assert len(candles) == 3
        assert candles[-1].close == Decimal("50009")

    @pytest.mark.asyncio()
    async def test_get_candles_filters_symbol(self) -> None:
        feed = BinanceWSFeed()
        msg = _make_kline_msg()
        feed._handle_message(msg)
        candles = await feed.get_candles("ETHUSDT")
        assert len(candles) == 0


class TestBinanceWSFeedConnectDisconnect:
    @pytest.mark.asyncio()
    async def test_subscribe_updates_url(self) -> None:
        feed = BinanceWSFeed()
        # Not running, so subscribe just updates internal state
        await feed.subscribe("ETHUSDT")
        assert feed._symbol == "ETHUSDT"
        assert "ethusdt" in feed._ws_url

    @pytest.mark.asyncio()
    async def test_disconnect_when_not_connected(self) -> None:
        feed = BinanceWSFeed()
        await feed.disconnect()
        assert feed.is_connected is False

    @pytest.mark.asyncio()
    async def test_connect_and_disconnect(self) -> None:
        """Test connection loop with a mocked websocket."""
        mock_ws = AsyncMock()
        mock_ws.__aiter__ = MagicMock(return_value=iter([]))

        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("src.data.binance_ws.websockets.connect", return_value=mock_ctx):
            feed = BinanceWSFeed()
            await feed.connect()
            # Give the loop a chance to start
            await asyncio.sleep(0.05)
            await feed.disconnect()
            assert feed.is_connected is False
