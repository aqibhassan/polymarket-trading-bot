"""Tests for MarketResolver."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.data.market_resolver import MarketResolver, ResolvedMarket


def _make_gamma_response(markets: list[dict]) -> httpx.Response:
    return httpx.Response(
        200,
        json=markets,
        request=httpx.Request("GET", "https://gamma-api.polymarket.com/markets"),
    )


@pytest.fixture()
def resolver() -> MarketResolver:
    return MarketResolver(cache_ttl=60)


def _btc_15m_market(
    market_id: str = "mkt-1",
    condition_id: str = "cond-1",
    question: str = "Will the BTC 15 minute candle be green?",
) -> dict:
    return {
        "id": market_id,
        "condition_id": condition_id,
        "question": question,
        "tokens": [
            {"outcome": "Yes", "token_id": "yes-tok-1"},
            {"outcome": "No", "token_id": "no-tok-1"},
        ],
    }


class TestMarketResolverFind:
    @pytest.mark.asyncio()
    async def test_finds_btc_15m_markets(self, resolver: MarketResolver) -> None:
        response = _make_gamma_response([_btc_15m_market()])
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.data.market_resolver.httpx.AsyncClient", return_value=mock_client):
            markets = await resolver.find_active_markets()

        assert len(markets) == 1
        m = markets[0]
        assert m.market_id == "mkt-1"
        assert m.condition_id == "cond-1"
        assert m.yes_token_id == "yes-tok-1"
        assert m.no_token_id == "no-tok-1"

    @pytest.mark.asyncio()
    async def test_filters_non_btc_markets(self, resolver: MarketResolver) -> None:
        markets_data = [
            _btc_15m_market(),
            {
                "id": "mkt-2",
                "condition_id": "cond-2",
                "question": "Will ETH go up?",
                "tokens": [
                    {"outcome": "Yes", "token_id": "y"},
                    {"outcome": "No", "token_id": "n"},
                ],
            },
        ]
        response = _make_gamma_response(markets_data)
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.data.market_resolver.httpx.AsyncClient", return_value=mock_client):
            results = await resolver.find_active_markets()

        assert len(results) == 1
        assert results[0].market_id == "mkt-1"

    @pytest.mark.asyncio()
    async def test_filters_non_15m_markets(self, resolver: MarketResolver) -> None:
        markets_data = [
            {
                "id": "mkt-2",
                "condition_id": "cond-2",
                "question": "Will BTC 1 hour candle be green?",
                "tokens": [
                    {"outcome": "Yes", "token_id": "y"},
                    {"outcome": "No", "token_id": "n"},
                ],
            },
        ]
        response = _make_gamma_response(markets_data)
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.data.market_resolver.httpx.AsyncClient", return_value=mock_client):
            results = await resolver.find_active_markets()

        assert len(results) == 0

    @pytest.mark.asyncio()
    async def test_skips_markets_without_tokens(self, resolver: MarketResolver) -> None:
        markets_data = [
            {
                "id": "mkt-3",
                "condition_id": "cond-3",
                "question": "Will the BTC 15 minute candle be green?",
                "tokens": [],
            },
        ]
        response = _make_gamma_response(markets_data)
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.data.market_resolver.httpx.AsyncClient", return_value=mock_client):
            results = await resolver.find_active_markets()

        assert len(results) == 0


class TestMarketResolverCaching:
    @pytest.mark.asyncio()
    async def test_cache_hit(self, resolver: MarketResolver) -> None:
        response = _make_gamma_response([_btc_15m_market()])
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.data.market_resolver.httpx.AsyncClient", return_value=mock_client):
            first = await resolver.find_active_markets()
            second = await resolver.find_active_markets()

        assert first == second
        # Only one HTTP call should have been made
        mock_client.get.assert_called_once()

    @pytest.mark.asyncio()
    async def test_cache_expired(self) -> None:
        resolver = MarketResolver(cache_ttl=0)  # Immediate expiry

        response = _make_gamma_response([_btc_15m_market()])
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.data.market_resolver.httpx.AsyncClient", return_value=mock_client):
            await resolver.find_active_markets()
            # Force cache expiry by manipulating timestamp
            resolver._cache_ts = time.monotonic() - 1
            await resolver.find_active_markets()

        assert mock_client.get.call_count == 2


class TestResolvedMarket:
    def test_frozen(self) -> None:
        m = ResolvedMarket(
            market_id="mkt-1",
            condition_id="cond-1",
            yes_token_id="yes",
            no_token_id="no",
            question="test",
        )
        with pytest.raises(AttributeError):
            m.market_id = "mkt-2"  # type: ignore[misc]
