"""Tests for PolymarketScanner."""

from __future__ import annotations

import time
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.data.polymarket_scanner import PolymarketScanner, find_best_btc_market


def _make_gamma_response(
    data: list[dict] | dict,
    url: str = "https://gamma-api.polymarket.com/events",
) -> httpx.Response:
    return httpx.Response(
        200,
        json=data,
        request=httpx.Request("GET", url),
    )


def _btc_15m_market(
    market_id: str = "mkt-1",
    condition_id: str = "cond-1",
    question: str = "Bitcoin Up or Down - February 8, 3:15PM-3:30PM ET",
    volume: str = "1000",
    outcome_prices: str = '["0.62","0.38"]',
    end_date: str = "2099-12-31T23:59:59Z",
    outcomes: str = '["Up", "Down"]',
    clob_token_ids: str = '["yes-tok-1", "no-tok-1"]',
) -> dict:
    return {
        "id": market_id,
        "conditionId": condition_id,
        "question": question,
        "volume": volume,
        "outcomePrices": outcome_prices,
        "endDate": end_date,
        "outcomes": outcomes,
        "clobTokenIds": clob_token_ids,
        "closed": False,
    }


def _btc_15m_market_legacy(
    market_id: str = "mkt-1",
    condition_id: str = "cond-1",
    question: str = "Will the BTC 15 minute candle be green?",
    volume: str = "1000",
    outcome_prices: str = '["0.62","0.38"]',
    end_date: str = "2099-12-31T23:59:59Z",
) -> dict:
    """Legacy format with tokens array and YES/NO outcomes."""
    return {
        "id": market_id,
        "conditionId": condition_id,
        "question": question,
        "volume": volume,
        "outcomePrices": outcome_prices,
        "endDate": end_date,
        "outcomes": '["Yes", "No"]',
        "tokens": [
            {"outcome": "Yes", "token_id": "yes-tok-1"},
            {"outcome": "No", "token_id": "no-tok-1"},
        ],
    }


def _wrap_in_event(markets: list[dict]) -> list[dict]:
    """Wrap market data in an events API response format."""
    return [{"slug": "btc-updown-15m-12345", "markets": markets}]


@pytest.fixture()
def scanner() -> PolymarketScanner:
    return PolymarketScanner(min_volume=500)


def _mock_httpx_client_for_events(
    event_response: list[dict],
) -> AsyncMock:
    """Create a mock httpx client that returns the same event response for all slug queries."""
    response = _make_gamma_response(event_response)
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get = AsyncMock(return_value=response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


def _mock_httpx_client(response: httpx.Response) -> AsyncMock:
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get = AsyncMock(return_value=response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


class TestScanActiveMarkets:
    @pytest.mark.asyncio()
    async def test_scan_finds_btc_markets(self, scanner: PolymarketScanner) -> None:
        event_data = _wrap_in_event([_btc_15m_market()])
        mock_client = _mock_httpx_client_for_events(event_data)

        with patch("src.data.polymarket_scanner.httpx.AsyncClient", return_value=mock_client):
            markets = await scanner.scan_active_markets()

        assert len(markets) == 1
        m = markets[0]
        assert m.condition_id == "cond-1"
        assert m.yes_token_id == "yes-tok-1"
        assert m.no_token_id == "no-tok-1"
        assert m.yes_price == Decimal("0.62")
        assert m.no_price == Decimal("0.38")
        assert "Up or Down" in m.question

    @pytest.mark.asyncio()
    async def test_scan_handles_up_down_outcomes(self, scanner: PolymarketScanner) -> None:
        """Markets with Up/Down outcomes should map to YES/NO correctly."""
        market = _btc_15m_market(
            outcomes='["Up", "Down"]',
            clob_token_ids='["up-token-123", "down-token-456"]',
            outcome_prices='["0.55", "0.45"]',
        )
        event_data = _wrap_in_event([market])
        mock_client = _mock_httpx_client_for_events(event_data)

        with patch("src.data.polymarket_scanner.httpx.AsyncClient", return_value=mock_client):
            markets = await scanner.scan_active_markets()

        assert len(markets) == 1
        m = markets[0]
        assert m.yes_token_id == "up-token-123"  # Up maps to YES
        assert m.no_token_id == "down-token-456"  # Down maps to NO
        assert m.yes_price == Decimal("0.55")
        assert m.no_price == Decimal("0.45")

    @pytest.mark.asyncio()
    async def test_scan_handles_legacy_format(self, scanner: PolymarketScanner) -> None:
        """Markets with legacy tokens array (YES/NO) should still work."""
        market = _btc_15m_market_legacy()
        event_data = _wrap_in_event([market])
        mock_client = _mock_httpx_client_for_events(event_data)

        with patch("src.data.polymarket_scanner.httpx.AsyncClient", return_value=mock_client):
            markets = await scanner.scan_active_markets()

        assert len(markets) == 1
        assert markets[0].yes_token_id == "yes-tok-1"
        assert markets[0].no_token_id == "no-tok-1"

    @pytest.mark.asyncio()
    async def test_scan_filters_low_liquidity(self, scanner: PolymarketScanner) -> None:
        """Markets with volume below min_volume should be filtered out."""
        low_vol_market = _btc_15m_market(volume="100")  # Below 500 threshold
        high_vol_market = _btc_15m_market(
            market_id="mkt-2",
            condition_id="cond-2",
            volume="2000",
        )
        event_data = _wrap_in_event([low_vol_market, high_vol_market])
        mock_client = _mock_httpx_client_for_events(event_data)

        with patch("src.data.polymarket_scanner.httpx.AsyncClient", return_value=mock_client):
            markets = await scanner.scan_active_markets()

        assert len(markets) == 1
        assert markets[0].condition_id == "cond-2"

    @pytest.mark.asyncio()
    async def test_scan_returns_empty_when_no_events(self, scanner: PolymarketScanner) -> None:
        """Should return empty list when no events are found for any slug."""
        mock_client = _mock_httpx_client_for_events([])  # Empty events

        with patch("src.data.polymarket_scanner.httpx.AsyncClient", return_value=mock_client):
            markets = await scanner.scan_active_markets()

        assert len(markets) == 0

    @pytest.mark.asyncio()
    async def test_scan_filters_non_15m_markets(self, scanner: PolymarketScanner) -> None:
        market = _btc_15m_market(question="Will the BTC 1 hour candle be green?")
        event_data = _wrap_in_event([market])
        mock_client = _mock_httpx_client_for_events(event_data)

        with patch("src.data.polymarket_scanner.httpx.AsyncClient", return_value=mock_client):
            markets = await scanner.scan_active_markets()

        assert len(markets) == 0

    @pytest.mark.asyncio()
    async def test_scan_filters_closed_markets(self, scanner: PolymarketScanner) -> None:
        """Closed markets should be filtered out."""
        market = _btc_15m_market()
        market["closed"] = True
        event_data = _wrap_in_event([market])
        mock_client = _mock_httpx_client_for_events(event_data)

        with patch("src.data.polymarket_scanner.httpx.AsyncClient", return_value=mock_client):
            markets = await scanner.scan_active_markets()

        assert len(markets) == 0

    @pytest.mark.asyncio()
    async def test_scan_handles_api_error(self, scanner: PolymarketScanner) -> None:
        """Should return empty list on API error when no cache available."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("connection failed"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.data.polymarket_scanner.httpx.AsyncClient", return_value=mock_client):
            markets = await scanner.scan_active_markets()

        assert len(markets) == 0

    @pytest.mark.asyncio()
    async def test_deduplicates_markets(self, scanner: PolymarketScanner) -> None:
        """Same market returned by multiple slugs should be deduplicated."""
        market = _btc_15m_market()
        # Mock returns same market for all 3 slug queries
        event_data = _wrap_in_event([market])
        mock_client = _mock_httpx_client_for_events(event_data)

        with patch("src.data.polymarket_scanner.httpx.AsyncClient", return_value=mock_client):
            markets = await scanner.scan_active_markets()

        assert len(markets) == 1

    @pytest.mark.asyncio()
    async def test_window_slugs_format(self) -> None:
        """Window slugs should follow the btc-updown-15m-{ts} format."""
        slugs = PolymarketScanner._window_slugs()
        assert len(slugs) == 3
        for slug in slugs:
            assert slug.startswith("btc-updown-15m-")
            ts_str = slug.replace("btc-updown-15m-", "")
            ts = int(ts_str)
            assert ts % 900 == 0  # Should be 15-minute aligned


class TestParseMarket:
    def test_parse_up_down_format(self, scanner: PolymarketScanner) -> None:
        """Test parsing current Polymarket BTC 15m format with Up/Down."""
        data = _btc_15m_market(
            outcomes='["Up", "Down"]',
            clob_token_ids='["up-tok", "down-tok"]',
            outcome_prices='["0.6", "0.4"]',
        )
        result = scanner._parse_market(data)
        assert result is not None
        assert result.yes_token_id == "up-tok"
        assert result.no_token_id == "down-tok"
        assert result.yes_price == Decimal("0.6")
        assert result.no_price == Decimal("0.4")

    def test_parse_rejects_no_tokens(self, scanner: PolymarketScanner) -> None:
        """Market without token IDs should be rejected."""
        data = _btc_15m_market(clob_token_ids="[]", outcomes="[]")
        # Also remove any tokens array
        data.pop("tokens", None)
        result = scanner._parse_market(data)
        assert result is None


class TestGetMarketPrices:
    @pytest.mark.asyncio()
    async def test_get_market_prices(self, scanner: PolymarketScanner) -> None:
        response = _make_gamma_response(
            {"outcomePrices": '["0.72","0.28"]'},
            url="https://gamma-api.polymarket.com/markets/mkt-1",
        )
        mock_client = _mock_httpx_client(response)

        with patch("src.data.polymarket_scanner.httpx.AsyncClient", return_value=mock_client):
            yes_price, no_price = await scanner.get_market_prices("mkt-1")

        assert yes_price == Decimal("0.72")
        assert no_price == Decimal("0.28")

    @pytest.mark.asyncio()
    async def test_get_market_prices_defaults_on_missing(self, scanner: PolymarketScanner) -> None:
        """Should return 0.5/0.5 when outcomePrices is missing."""
        response = _make_gamma_response(
            {},
            url="https://gamma-api.polymarket.com/markets/mkt-1",
        )
        mock_client = _mock_httpx_client(response)

        with patch("src.data.polymarket_scanner.httpx.AsyncClient", return_value=mock_client):
            yes_price, no_price = await scanner.get_market_prices("mkt-1")

        assert yes_price == Decimal("0.5")
        assert no_price == Decimal("0.5")


class TestCaching:
    @pytest.mark.asyncio()
    async def test_caching_prevents_duplicate_calls(self, scanner: PolymarketScanner) -> None:
        """Second call within 5s TTL should use cache, not make another HTTP call."""
        event_data = _wrap_in_event([_btc_15m_market()])
        mock_client = _mock_httpx_client_for_events(event_data)

        with patch("src.data.polymarket_scanner.httpx.AsyncClient", return_value=mock_client):
            first = await scanner.scan_active_markets()
            second = await scanner.scan_active_markets()

        assert first == second
        # 3 calls for first scan (one per slug), 0 for cached second
        assert mock_client.get.call_count == 3

    @pytest.mark.asyncio()
    async def test_cache_expires(self) -> None:
        """After cache TTL expires, a new HTTP call should be made."""
        scanner = PolymarketScanner(min_volume=500)
        event_data = _wrap_in_event([_btc_15m_market()])
        mock_client = _mock_httpx_client_for_events(event_data)

        with patch("src.data.polymarket_scanner.httpx.AsyncClient", return_value=mock_client):
            await scanner.scan_active_markets()
            # Force cache expiry
            scanner._scan_cache_ts = time.monotonic() - 10
            await scanner.scan_active_markets()

        # 3 calls per scan (one per slug) x 2 scans = 6
        assert mock_client.get.call_count == 6

    @pytest.mark.asyncio()
    async def test_price_caching(self, scanner: PolymarketScanner) -> None:
        """Price lookups should be cached within TTL."""
        response = _make_gamma_response(
            {"outcomePrices": '["0.72","0.28"]'},
            url="https://gamma-api.polymarket.com/markets/mkt-1",
        )
        mock_client = _mock_httpx_client(response)

        with patch("src.data.polymarket_scanner.httpx.AsyncClient", return_value=mock_client):
            first = await scanner.get_market_prices("mkt-1")
            second = await scanner.get_market_prices("mkt-1")

        assert first == second
        mock_client.get.assert_called_once()


class TestGetMarketOrderbook:
    @pytest.mark.asyncio()
    async def test_get_market_orderbook(self, scanner: PolymarketScanner) -> None:
        response = _make_gamma_response(
            {
                "bids": [
                    {"price": "0.55", "size": "100"},
                    {"price": "0.54", "size": "200"},
                ],
                "asks": [
                    {"price": "0.57", "size": "150"},
                ],
            },
            url="https://clob.polymarket.com/book",
        )
        mock_client = _mock_httpx_client(response)

        with patch("src.data.polymarket_scanner.httpx.AsyncClient", return_value=mock_client):
            book = await scanner.get_market_orderbook("token-123")

        assert len(book.bids) == 2
        assert len(book.asks) == 1
        assert book.bids[0].price == Decimal("0.55")
        assert book.asks[0].size == Decimal("150")
        assert book.market_id == "token-123"


class TestFindBestBtcMarket:
    @pytest.mark.asyncio()
    async def test_returns_best_market(self) -> None:
        markets = [
            _btc_15m_market(market_id="mkt-1", condition_id="cond-1", volume="1000"),
            _btc_15m_market(
                market_id="mkt-2",
                condition_id="cond-2",
                volume="5000",
                end_date="2099-12-31T23:59:59Z",
            ),
        ]
        event_data = _wrap_in_event(markets)
        mock_client = _mock_httpx_client_for_events(event_data)

        with patch("src.data.polymarket_scanner.httpx.AsyncClient", return_value=mock_client):
            result = await find_best_btc_market(min_volume=500)

        assert result is not None

    @pytest.mark.asyncio()
    async def test_returns_none_when_no_markets(self) -> None:
        mock_client = _mock_httpx_client_for_events([])

        with patch("src.data.polymarket_scanner.httpx.AsyncClient", return_value=mock_client):
            result = await find_best_btc_market(min_volume=500)

        assert result is None
