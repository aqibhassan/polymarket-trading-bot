"""Tests for CircuitBreaker — state transitions, auto-recovery."""

from __future__ import annotations

import time

from src.execution.circuit_breaker import CircuitBreaker


def test_initial_state_closed() -> None:
    cb = CircuitBreaker(max_failures=3, cooldown_seconds=10)
    assert cb.state == "CLOSED"
    assert cb.can_execute() is True


def test_stays_closed_under_threshold() -> None:
    cb = CircuitBreaker(max_failures=3)
    cb.record_failure()
    cb.record_failure()
    assert cb.state == "CLOSED"
    assert cb.can_execute() is True


def test_opens_at_threshold() -> None:
    cb = CircuitBreaker(max_failures=3)
    for _ in range(3):
        cb.record_failure()
    assert cb.state == "OPEN"
    assert cb.can_execute() is False


def test_success_resets_count() -> None:
    cb = CircuitBreaker(max_failures=3)
    cb.record_failure()
    cb.record_failure()
    cb.record_success()
    assert cb.state == "CLOSED"
    # Should need 3 more failures to open
    cb.record_failure()
    cb.record_failure()
    assert cb.state == "CLOSED"


def test_auto_recovery_half_open() -> None:
    cb = CircuitBreaker(max_failures=2, cooldown_seconds=0.1)
    cb.record_failure()
    cb.record_failure()
    assert cb.state == "OPEN"
    time.sleep(0.15)
    assert cb.state == "HALF_OPEN"
    assert cb.can_execute() is True


def test_half_open_success_closes() -> None:
    cb = CircuitBreaker(max_failures=2, cooldown_seconds=0.1)
    cb.record_failure()
    cb.record_failure()
    time.sleep(0.15)
    assert cb.state == "HALF_OPEN"
    cb.record_success()
    assert cb.state == "CLOSED"


def test_half_open_failure_reopens() -> None:
    cb = CircuitBreaker(max_failures=1, cooldown_seconds=0.1)
    cb.record_failure()
    assert cb.state == "OPEN"
    time.sleep(0.15)
    assert cb.state == "HALF_OPEN"
    cb.record_failure()
    assert cb.state == "OPEN"


def test_default_config() -> None:
    cb = CircuitBreaker()
    # Should need 5 failures to open
    for _ in range(4):
        cb.record_failure()
    assert cb.state == "CLOSED"
    cb.record_failure()
    assert cb.state == "OPEN"
