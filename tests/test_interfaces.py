"""Tests for Protocol interfaces."""

from __future__ import annotations

from decimal import Decimal

from src.interfaces import RiskDecision


class TestRiskDecision:
    def test_approved_decision(self) -> None:
        decision = RiskDecision(approved=True, reason="all checks passed", max_size=Decimal("100"))
        assert decision.approved is True
        assert decision.reason == "all checks passed"
        assert decision.max_size == Decimal("100")

    def test_rejected_decision(self) -> None:
        decision = RiskDecision(approved=False, reason="exceeds max position")
        assert decision.approved is False
        assert decision.reason == "exceeds max position"
        assert decision.max_size == Decimal("0")
