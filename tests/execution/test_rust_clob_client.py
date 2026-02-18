"""Tests for the Rust CLOB client wrapper with Python fallback."""

from __future__ import annotations

import pytest

from src.execution.rust_clob_client import RUST_AVAILABLE, ClobClientWrapper


class TestRustAvailableFlag:
    """Test that the availability flag is set correctly."""

    def test_rust_available_is_bool(self) -> None:
        assert isinstance(RUST_AVAILABLE, bool)


class TestClobClientWrapperFallback:
    """Test the Python fallback path (no Rust extension needed)."""

    def test_wrapper_has_expected_methods(self) -> None:
        """Verify the wrapper exposes all required async methods."""
        methods = [
            "create_and_post_order",
            "get_order_book",
            "cancel",
            "get_order",
            "get_orders",
            "get_balance_allowance",
            "get_midpoint",
        ]
        for method in methods:
            assert hasattr(ClobClientWrapper, method), f"Missing method: {method}"


class TestGetOrderStatusBridge:
    """Test that get_order_status is properly exposed on the bridge."""

    @pytest.mark.asyncio
    async def test_bridge_get_order_status_no_live_trader(self) -> None:
        """Bridge without live trader returns UNKNOWN."""
        from src.execution.bridge import ExecutionBridge
        from src.execution.paper_trader import PaperTrader
        from src.models.order import OrderSide

        paper = PaperTrader(initial_balance=10000)
        bridge = ExecutionBridge(mode="paper", paper_trader=paper)

        result = await bridge.get_order_status("some-order-id")
        assert result["status"] == "UNKNOWN"

    @pytest.mark.asyncio
    async def test_bridge_get_order_status_with_mock_live_trader(self) -> None:
        """Bridge with mock live trader delegates correctly."""
        from unittest.mock import AsyncMock, MagicMock

        from src.execution.bridge import ExecutionBridge

        mock_trader = MagicMock()
        mock_trader.get_order_status = AsyncMock(
            return_value={"status": "MATCHED", "size_matched": "10"}
        )
        mock_trader.submit_order = AsyncMock()
        mock_trader.cancel_order = AsyncMock()
        mock_trader.get_order = AsyncMock()
        mock_trader.mode = "live"

        bridge = ExecutionBridge(
            mode="live",
            live_trader=mock_trader,
            skip_confirmation=True,
        )

        result = await bridge.get_order_status("0xabc123")
        assert result["status"] == "MATCHED"
        assert result["size_matched"] == "10"
        mock_trader.get_order_status.assert_awaited_once_with("0xabc123")
