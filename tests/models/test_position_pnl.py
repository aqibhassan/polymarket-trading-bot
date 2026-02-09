"""Tests for Position.unrealized_pnl method."""

from __future__ import annotations

from decimal import Decimal

from src.models.market import Position  # noqa: TCH001


class TestPositionUnrealizedPnl:
    def test_unrealized_profit(self, sample_position: Position) -> None:
        pnl = sample_position.unrealized_pnl(Decimal("0.50"))
        assert pnl == Decimal("10.00")  # (0.50 - 0.40) * 100

    def test_unrealized_loss(self, sample_position: Position) -> None:
        pnl = sample_position.unrealized_pnl(Decimal("0.35"))
        assert pnl == Decimal("-5.00")  # (0.35 - 0.40) * 100

    def test_unrealized_breakeven(self, sample_position: Position) -> None:
        pnl = sample_position.unrealized_pnl(Decimal("0.40"))
        assert pnl == Decimal("0")

    def test_unrealized_pnl_returns_decimal(self, sample_position: Position) -> None:
        result = sample_position.unrealized_pnl(Decimal("0.45"))
        assert isinstance(result, Decimal)
