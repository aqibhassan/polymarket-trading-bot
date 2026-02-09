"""Tests for PolymarketLiveTrader — signing, submission, rate limiting, error handling."""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.execution.polymarket_signer import (
    PolymarketLiveTrader,
    _mask_secret,
)
from src.models.order import OrderSide, OrderStatus, OrderType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_KEY = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbb"
_FAKE_FUNDER = "0x1234567890abcdef1234567890abcdef12345678"


def _env_vars() -> dict[str, str]:
    return {
        "POLYMARKET_PRIVATE_KEY": _FAKE_KEY,
        "POLYMARKET_FUNDER_ADDRESS": _FAKE_FUNDER,
    }


def _mock_clob_client() -> MagicMock:
    """Build a mock ClobClient with default return values."""
    mock = MagicMock()
    mock.create_or_derive_api_creds.return_value = MagicMock()
    mock.set_api_creds.return_value = None
    mock.create_order.return_value = MagicMock()
    mock.post_order.return_value = {"orderID": "exchange-order-123"}
    mock.cancel.return_value = {"success": True}
    mock.get_positions.return_value = [
        {"token_id": "t1", "size": "100"},
    ]
    mock.get_balance.return_value = "5000.00"
    return mock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_clob() -> MagicMock:
    return _mock_clob_client()


@pytest.fixture(autouse=True)
def _patch_clob_imports() -> Any:
    """Patch the deferred py-clob-client imports so tests don't need the real library."""
    mock_order_args = MagicMock()
    mock_order_type = MagicMock()
    with patch("src.execution.polymarket_signer._create_order_args", return_value=mock_order_args), \
         patch("src.execution.polymarket_signer._clob_order_type", return_value=mock_order_type):
        yield


@pytest.fixture
def trader(mock_clob: MagicMock) -> PolymarketLiveTrader:
    """Create a PolymarketLiveTrader with injected mock ClobClient."""
    with patch.dict("os.environ", _env_vars()):
        t = PolymarketLiveTrader(
            clob_url="https://clob.polymarket.com",
            chain_id=137,
            _clob_client=mock_clob,
        )
    return t


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


class TestInit:
    def test_missing_private_key_raises(self) -> None:
        with patch.dict("os.environ", {"POLYMARKET_PRIVATE_KEY": ""}, clear=False), \
             pytest.raises(ValueError, match="POLYMARKET_PRIVATE_KEY"):
            PolymarketLiveTrader()

    def test_mode_is_live(self, trader: PolymarketLiveTrader) -> None:
        assert trader.mode == "live"

    def test_injected_client_used(self, trader: PolymarketLiveTrader, mock_clob: MagicMock) -> None:
        # When _clob_client is injected, create_or_derive_api_creds is NOT called
        # (only called when constructing a real ClobClient)
        mock_clob.create_or_derive_api_creds.assert_not_called()
        assert trader._clob_client is mock_clob


# ---------------------------------------------------------------------------
# Secret masking
# ---------------------------------------------------------------------------


class TestMasking:
    def test_mask_long_key(self) -> None:
        assert _mask_secret("abcdefgh") == "****efgh"

    def test_mask_short_key(self) -> None:
        assert _mask_secret("ab") == "****"

    def test_mask_exactly_4(self) -> None:
        assert _mask_secret("abcd") == "****"

    def test_mask_5_chars(self) -> None:
        assert _mask_secret("abcde") == "*bcde"


# ---------------------------------------------------------------------------
# Order submission
# ---------------------------------------------------------------------------


class TestSubmitOrder:
    @pytest.mark.asyncio
    async def test_submit_buy_success(self, trader: PolymarketLiveTrader, mock_clob: MagicMock) -> None:
        order = await trader.submit_order(
            market_id="m1",
            token_id="t1",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=Decimal("0.55"),
            size=Decimal("100"),
            strategy_id="strat_1",
        )
        assert order.status == OrderStatus.SUBMITTED
        assert order.exchange_order_id == "exchange-order-123"
        mock_clob.create_order.assert_called_once()
        mock_clob.post_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_submit_sell_success(self, trader: PolymarketLiveTrader, mock_clob: MagicMock) -> None:
        order = await trader.submit_order(
            market_id="m1",
            token_id="t1",
            side=OrderSide.SELL,
            order_type=OrderType.GTC,
            price=Decimal("0.60"),
            size=Decimal("50"),
        )
        assert order.status == OrderStatus.SUBMITTED

    @pytest.mark.asyncio
    async def test_submit_exchange_error(self, trader: PolymarketLiveTrader, mock_clob: MagicMock) -> None:
        mock_clob.post_order.side_effect = Exception("insufficient balance")
        order = await trader.submit_order(
            market_id="m1",
            token_id="t1",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=Decimal("0.55"),
            size=Decimal("100"),
        )
        assert order.status == OrderStatus.REJECTED

    @pytest.mark.asyncio
    async def test_submit_no_order_id_in_response(self, trader: PolymarketLiveTrader, mock_clob: MagicMock) -> None:
        mock_clob.post_order.return_value = {"errorMsg": "invalid price"}
        order = await trader.submit_order(
            market_id="m1",
            token_id="t1",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=Decimal("1.50"),
            size=Decimal("10"),
        )
        assert order.status == OrderStatus.REJECTED

    @pytest.mark.asyncio
    async def test_submit_unexpected_response_type(self, trader: PolymarketLiveTrader, mock_clob: MagicMock) -> None:
        mock_clob.post_order.return_value = "not a dict"
        order = await trader.submit_order(
            market_id="m1",
            token_id="t1",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=Decimal("0.50"),
            size=Decimal("10"),
        )
        assert order.status == OrderStatus.REJECTED

    @pytest.mark.asyncio
    async def test_order_tracked_internally(self, trader: PolymarketLiveTrader) -> None:
        order = await trader.submit_order(
            market_id="m1",
            token_id="t1",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=Decimal("0.55"),
            size=Decimal("100"),
        )
        found = await trader.get_order(str(order.id))
        assert found is not None
        assert found.id == order.id


# ---------------------------------------------------------------------------
# Cancel order
# ---------------------------------------------------------------------------


class TestCancelOrder:
    @pytest.mark.asyncio
    async def test_cancel_success(self, trader: PolymarketLiveTrader, mock_clob: MagicMock) -> None:
        order = await trader.submit_order(
            market_id="m1",
            token_id="t1",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=Decimal("0.55"),
            size=Decimal("100"),
        )
        result = await trader.cancel_order(str(order.id))
        assert result is True
        cancelled = await trader.get_order(str(order.id))
        assert cancelled is not None
        assert cancelled.status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_nonexistent(self, trader: PolymarketLiveTrader) -> None:
        result = await trader.cancel_order("nonexistent-id")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_exchange_error(self, trader: PolymarketLiveTrader, mock_clob: MagicMock) -> None:
        order = await trader.submit_order(
            market_id="m1",
            token_id="t1",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=Decimal("0.55"),
            size=Decimal("100"),
        )
        mock_clob.cancel.side_effect = Exception("network error")
        result = await trader.cancel_order(str(order.id))
        assert result is False


# ---------------------------------------------------------------------------
# Positions and balance
# ---------------------------------------------------------------------------


class TestPositionsAndBalance:
    @pytest.mark.asyncio
    async def test_get_positions(self, trader: PolymarketLiveTrader, mock_clob: MagicMock) -> None:
        positions = await trader.get_positions()
        assert len(positions) == 1
        assert positions[0]["token_id"] == "t1"
        mock_clob.get_positions.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_positions_error(self, trader: PolymarketLiveTrader, mock_clob: MagicMock) -> None:
        mock_clob.get_positions.side_effect = Exception("api error")
        positions = await trader.get_positions()
        assert positions == []

    @pytest.mark.asyncio
    async def test_get_balance(self, trader: PolymarketLiveTrader, mock_clob: MagicMock) -> None:
        balance = await trader.get_balance()
        assert balance == Decimal("5000.00")
        mock_clob.get_balance.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_balance_error(self, trader: PolymarketLiveTrader, mock_clob: MagicMock) -> None:
        mock_clob.get_balance.side_effect = Exception("api error")
        balance = await trader.get_balance()
        assert balance == Decimal("0")


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    @pytest.mark.asyncio
    async def test_rate_limiter_invoked(self, trader: PolymarketLiveTrader) -> None:
        """Verify the rate limiter is used (tokens decrease after calls)."""
        initial_tokens = trader._rate_limiter.tokens_remaining
        await trader.submit_order(
            market_id="m1",
            token_id="t1",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=Decimal("0.55"),
            size=Decimal("100"),
        )
        # At least one token consumed
        assert trader._rate_limiter.tokens_remaining < initial_tokens

    @pytest.mark.asyncio
    async def test_rate_limiter_on_cancel(self, trader: PolymarketLiveTrader, mock_clob: MagicMock) -> None:
        """Verify rate limiter is used for cancel calls too."""
        order = await trader.submit_order(
            market_id="m1",
            token_id="t1",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=Decimal("0.55"),
            size=Decimal("100"),
        )
        tokens_before = trader._rate_limiter.tokens_remaining
        await trader.cancel_order(str(order.id))
        assert trader._rate_limiter.tokens_remaining < tokens_before

    @pytest.mark.asyncio
    async def test_rate_limiter_on_get_balance(self, trader: PolymarketLiveTrader) -> None:
        tokens_before = trader._rate_limiter.tokens_remaining
        await trader.get_balance()
        assert trader._rate_limiter.tokens_remaining < tokens_before

    @pytest.mark.asyncio
    async def test_rate_limiter_on_get_positions(self, trader: PolymarketLiveTrader) -> None:
        tokens_before = trader._rate_limiter.tokens_remaining
        await trader.get_positions()
        assert trader._rate_limiter.tokens_remaining < tokens_before


# ---------------------------------------------------------------------------
# Order type mapping (mocked — no real py_clob_client import)
# ---------------------------------------------------------------------------


class TestOrderFields:
    @pytest.mark.asyncio
    async def test_order_fields_preserved(self, trader: PolymarketLiveTrader) -> None:
        order = await trader.submit_order(
            market_id="m1",
            token_id="t1",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=Decimal("0.55"),
            size=Decimal("100"),
            strategy_id="test_strat",
        )
        assert order.market_id == "m1"
        assert order.token_id == "t1"
        assert order.side == OrderSide.BUY
        assert order.price == Decimal("0.55")
        assert order.size == Decimal("100")
        assert order.strategy_id == "test_strat"

    @pytest.mark.asyncio
    async def test_get_nonexistent_order(self, trader: PolymarketLiveTrader) -> None:
        assert await trader.get_order("does-not-exist") is None
