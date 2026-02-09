"""Tests for structured logging."""

from __future__ import annotations

from src.core.logging import get_audit_logger, get_logger, log_order_event


class TestStructuredLogging:
    def test_get_logger_returns_bound_logger(self) -> None:
        logger = get_logger("test.module")
        assert logger is not None

    def test_get_logger_same_name_returns_logger(self) -> None:
        logger1 = get_logger("test.same")
        logger2 = get_logger("test.same")
        assert logger1 is not None
        assert logger2 is not None

    def test_get_audit_logger(self) -> None:
        logger = get_audit_logger()
        assert logger is not None

    def test_log_order_event_does_not_raise(self) -> None:
        log_order_event(
            action="submit",
            order_id="test-order-123",
            price="0.55",
            size="100",
        )

    def test_log_order_event_with_rejection(self) -> None:
        log_order_event(
            action="reject",
            order_id="test-order-456",
            reason="oversized position",
        )
