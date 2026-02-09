"""Circuit breaker — halts execution after consecutive failures."""

from __future__ import annotations

import time
from enum import Enum

from src.core.logging import get_logger

logger = get_logger(__name__)


class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """Halts execution after N consecutive failures with auto-recovery.

    States:
        CLOSED  — normal operation
        OPEN    — halted, no executions allowed
        HALF_OPEN — cooldown elapsed, allow one trial execution
    """

    def __init__(
        self,
        max_failures: int = 5,
        cooldown_seconds: float = 60,
    ) -> None:
        self._max_failures = max_failures
        self._cooldown_seconds = cooldown_seconds
        self._failure_count = 0
        self._state = CircuitState.CLOSED
        self._opened_at: float | None = None

    @property
    def state(self) -> str:
        self._check_recovery()
        return self._state.value

    def can_execute(self) -> bool:
        self._check_recovery()
        return self._state != CircuitState.OPEN

    def record_success(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            logger.info("circuit_breaker_closed", previous_failures=self._failure_count)
        self._failure_count = 0
        self._state = CircuitState.CLOSED
        self._opened_at = None

    def record_failure(self) -> None:
        self._failure_count += 1
        if self._failure_count >= self._max_failures:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()
            logger.warning(
                "circuit_breaker_opened",
                failure_count=self._failure_count,
                cooldown=self._cooldown_seconds,
            )

    def _check_recovery(self) -> None:
        if (
            self._state == CircuitState.OPEN
            and self._opened_at is not None
            and time.monotonic() - self._opened_at >= self._cooldown_seconds
        ):
            self._state = CircuitState.HALF_OPEN
            logger.info("circuit_breaker_half_open", cooldown_elapsed=self._cooldown_seconds)
