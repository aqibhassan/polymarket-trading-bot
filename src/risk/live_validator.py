"""Live performance validator — auto-halts trading on degraded metrics.

Compares rolling live metrics against configurable thresholds.
When metrics degrade past thresholds, signals the bot to halt trading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""

    should_halt: bool
    reason: str | None = None
    win_rate: float | None = None
    brier_score: float | None = None
    trade_count: int = 0


@dataclass
class TradeOutcome:
    """Record of a single trade's predicted vs actual outcome."""

    trade_id: str
    predicted_prob: float  # calibrated posterior at entry
    won: bool
    entry_price: float
    pnl: float


class LiveValidator:
    """Validates live trading performance and triggers auto-halt.

    Tracks rolling trade outcomes and checks against thresholds:
    - Win rate below halt_win_rate → halt
    - Brier score above halt_brier_score → halt

    Checks are only performed after min_trades outcomes are recorded,
    and re-checked every check_interval trades.
    """

    def __init__(
        self,
        *,
        min_trades: int = 20,
        halt_win_rate: float = 0.45,
        halt_brier_score: float = 0.30,
        check_interval: int = 5,
    ) -> None:
        self._min_trades = min_trades
        self._halt_win_rate = halt_win_rate
        self._halt_brier_score = halt_brier_score
        self._check_interval = check_interval
        self._outcomes: list[TradeOutcome] = []
        self._halted = False
        self._halt_reason: str | None = None

    @property
    def is_halted(self) -> bool:
        return self._halted

    @property
    def halt_reason(self) -> str | None:
        return self._halt_reason

    @property
    def trade_count(self) -> int:
        return len(self._outcomes)

    def record_outcome(
        self,
        trade_id: str,
        predicted_prob: float,
        won: bool,
        entry_price: float = 0.0,
        pnl: float = 0.0,
    ) -> None:
        """Record a completed trade outcome."""
        self._outcomes.append(TradeOutcome(
            trade_id=trade_id,
            predicted_prob=predicted_prob,
            won=won,
            entry_price=entry_price,
            pnl=pnl,
        ))

    def should_halt(self) -> ValidationResult:
        """Check if trading should be halted based on rolling metrics.

        Only checks after min_trades outcomes, and respects check_interval.
        Once halted, stays halted (requires manual reset).
        """
        n = len(self._outcomes)

        if self._halted:
            return ValidationResult(
                should_halt=True,
                reason=self._halt_reason,
                trade_count=n,
            )

        if n < self._min_trades:
            return ValidationResult(should_halt=False, trade_count=n)

        if n % self._check_interval != 0:
            return ValidationResult(should_halt=False, trade_count=n)

        wins = sum(1 for o in self._outcomes if o.won)
        win_rate = wins / n if n > 0 else 0.0

        # Brier score: mean squared error of probability predictions
        brier = sum(
            (o.predicted_prob - (1.0 if o.won else 0.0)) ** 2
            for o in self._outcomes
        ) / n

        result = ValidationResult(
            should_halt=False,
            win_rate=win_rate,
            brier_score=brier,
            trade_count=n,
        )

        if win_rate < self._halt_win_rate:
            result.should_halt = True
            result.reason = (
                f"Win rate {win_rate:.1%} below threshold {self._halt_win_rate:.1%} "
                f"over {n} trades"
            )
            self._halted = True
            self._halt_reason = result.reason
            log.warning(
                "live_validator_halt",
                reason=result.reason,
                win_rate=round(win_rate, 4),
                trades=n,
            )

        elif brier > self._halt_brier_score:
            result.should_halt = True
            result.reason = (
                f"Brier score {brier:.3f} exceeds threshold {self._halt_brier_score:.3f} "
                f"over {n} trades"
            )
            self._halted = True
            self._halt_reason = result.reason
            log.warning(
                "live_validator_halt",
                reason=result.reason,
                brier_score=round(brier, 4),
                trades=n,
            )

        return result

    def reset(self) -> None:
        """Reset halt state (manual override). Does not clear history."""
        self._halted = False
        self._halt_reason = None
        log.info("live_validator_reset", trades=len(self._outcomes))

    def get_stats(self) -> dict[str, Any]:
        """Return current rolling statistics."""
        n = len(self._outcomes)
        if n == 0:
            return {"trade_count": 0}

        wins = sum(1 for o in self._outcomes if o.won)
        brier = sum(
            (o.predicted_prob - (1.0 if o.won else 0.0)) ** 2
            for o in self._outcomes
        ) / n
        total_pnl = sum(o.pnl for o in self._outcomes)

        return {
            "trade_count": n,
            "win_rate": round(wins / n, 4),
            "brier_score": round(brier, 4),
            "total_pnl": round(total_pnl, 2),
            "halted": self._halted,
            "halt_reason": self._halt_reason,
        }
