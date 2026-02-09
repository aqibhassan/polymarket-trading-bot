"""Tests for PolymarketClient."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.data.polymarket_client import PolymarketClient, _mask_key


class TestMaskKey:
    def test_mask_long_key(self) -> None:
        assert _mask_key("abcdefgh1234") == "********1234"

    def test_mask_short_key(self) -> None:
        assert _mask_key("abc") == "****"

    def test_mask_exactly_four(self) -> None:
        assert _mask_key("abcd") == "****"


class TestPolymarketClientInit:
    def test_default_init(self) -> None:
        client = PolymarketClient()
        assert client._clob_url == "https://clob.polymarket.com"

    def test_custom_url(self) -> None:
        client = PolymarketClient(clob_url="https://custom.api/")
        assert client._clob_url == "https://custom.api"

    def test_env_vars_read(self) -> None:
        with patch.dict("os.environ", {
            "POLYMARKET_API_KEY": "test-key-1234",
            "POLYMARKET_SECRET": "test-secret",
            "POLYMARKET_PASSPHRASE": "test-pass",
        }):
            client = PolymarketClient()
            assert client._api_key == "test-key-1234"
            assert client._secret == "test-secret"
            assert client._passphrase == "test-pass"


class TestPolymarketClientGetMarket:
    @pytest.mark.asyncio()
    async def test_get_market_success(self) -> None:
        mock_response = httpx.Response(
            200,
            json={
                "condition_id": "cond-123",
                "question": "Will BTC go up?",
                "seconds_until_close": 600,
                "tokens": [
                    {"outcome": "Yes", "token_id": "yes-tok", "price": "0.62"},
                    {"outcome": "No", "token_id": "no-tok", "price": "0.38"},
                ],
            },
            request=httpx.Request("GET", "https://test.com"),
        )
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False

        client = PolymarketClient()
        client._client = mock_client

        state = await client.get_market("cond-123")
        assert state.condition_id == "cond-123"
        assert state.yes_token_id == "yes-tok"
        assert state.no_token_id == "no-tok"
        assert state.yes_price == Decimal("0.62")
        assert state.no_price == Decimal("0.38")
        assert state.time_remaining_seconds == 600

    @pytest.mark.asyncio()
    async def test_get_market_http_error(self) -> None:
        mock_response = httpx.Response(
            404,
            request=httpx.Request("GET", "https://test.com"),
        )
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False

        client = PolymarketClient()
        client._client = mock_client

        with pytest.raises(httpx.HTTPStatusError):
            await client.get_market("nonexistent")


class TestPolymarketClientGetOrderbook:
    @pytest.mark.asyncio()
    async def test_get_orderbook_success(self) -> None:
        mock_response = httpx.Response(
            200,
            json={
                "bids": [
                    {"price": "0.55", "size": "100"},
                    {"price": "0.54", "size": "200"},
                ],
                "asks": [
                    {"price": "0.57", "size": "150"},
                ],
            },
            request=httpx.Request("GET", "https://test.com"),
        )
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False

        client = PolymarketClient()
        client._client = mock_client

        book = await client.get_orderbook("token-123")
        assert len(book.bids) == 2
        assert len(book.asks) == 1
        assert book.bids[0].price == Decimal("0.55")
        assert book.asks[0].size == Decimal("150")


class TestPolymarketClientPlaceOrder:
    @pytest.mark.asyncio()
    async def test_place_order(self) -> None:
        mock_response = httpx.Response(
            200,
            json={"orderID": "order-abc", "status": "LIVE"},
            request=httpx.Request("POST", "https://test.com"),
        )
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False

        client = PolymarketClient()
        client._client = mock_client

        result = await client.place_order(
            token_id="token-123",
            side="BUY",
            price=Decimal("0.55"),
            size=Decimal("100"),
        )
        assert result["orderID"] == "order-abc"
        mock_client.post.assert_called_once()


class TestPolymarketClientCancelOrder:
    @pytest.mark.asyncio()
    async def test_cancel_order_success(self) -> None:
        mock_response = httpx.Response(
            200,
            request=httpx.Request("DELETE", "https://test.com"),
        )
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.delete = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False

        client = PolymarketClient()
        client._client = mock_client

        result = await client.cancel_order("order-abc")
        assert result is True

    @pytest.mark.asyncio()
    async def test_cancel_order_not_found(self) -> None:
        mock_response = httpx.Response(
            404,
            request=httpx.Request("DELETE", "https://test.com"),
        )
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.delete = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False

        client = PolymarketClient()
        client._client = mock_client

        result = await client.cancel_order("nonexistent")
        assert result is False


class TestPolymarketClientGetBalance:
    @pytest.mark.asyncio()
    async def test_get_balance(self) -> None:
        mock_response = httpx.Response(
            200,
            json={"balance": "1234.56"},
            request=httpx.Request("GET", "https://test.com"),
        )
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False

        client = PolymarketClient()
        client._client = mock_client

        balance = await client.get_balance()
        assert balance == Decimal("1234.56")


class TestPolymarketClientClose:
    @pytest.mark.asyncio()
    async def test_close(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False

        client = PolymarketClient()
        client._client = mock_client

        await client.close()
        mock_client.aclose.assert_called_once()
        assert client._client is None
