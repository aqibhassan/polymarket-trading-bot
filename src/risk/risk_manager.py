"""Pre-trade risk gate — validates every order before execution."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from src.core.logging import get_logger
from src.interfaces import RiskDecision
from src.risk.kill_switch import KillSwitch

if TYPE_CHECKING:
    from src.models.signal import Signal

log = get_logger(__name__)


class RiskManager:
    """Pre-trade risk gate implementing the RiskGate protocol.

    Checks every order against risk limits before allowing execution.
    """

    def __init__(
        self,
        max_position_pct: Decimal = Decimal("0.02"),
        max_daily_drawdown_pct: Decimal = Decimal("0.05"),
        max_order_book_pct: Decimal = Decimal("0.10"),
        kill_switch: KillSwitch | None = None,
        balance_floor_pct: Decimal = Decimal("0"),
        initial_balance: Decimal = Decimal("0"),
    ) -> None:
        self._max_position_pct = max_position_pct
        self._max_daily_drawdown_pct = max_daily_drawdown_pct
        self._max_order_book_pct = max_order_book_pct
        self._kill_switch = kill_switch or KillSwitch(max_daily_drawdown_pct)
        self._balance_floor_pct = balance_floor_pct
        self._initial_balance = initial_balance

    def has_stop_loss(self, signal: Signal) -> bool:
        """Check whether the signal has a stop-loss set."""
        return signal.stop_loss is not None

    def check_order(
        self,
        signal: Signal,
        position_size: Decimal,
        current_drawdown: Decimal,
        balance: Decimal = Decimal("0"),
        book_depth: Decimal = Decimal("0"),
        estimated_fee: Decimal = Decimal("0"),
    ) -> RiskDecision:
        """Validate an order against all risk checks.

        Args:
            signal: The trading signal to validate.
            position_size: Proposed position size in notional terms.
            current_drawdown: Current daily drawdown as a fraction (e.g. 0.03 = 3%).
            balance: Current account balance for sizing checks.
            book_depth: Total order book depth for impact checks.
            estimated_fee: Estimated fee cost for the order.

        Returns:
            RiskDecision with approved=True if all checks pass,
            or approved=False with the rejection reason.
        """
        # 0. Zero/negative balance — reject immediately
        if balance <= 0:
            reason = "zero or negative balance — cannot trade"
            log.warning("risk_rejected", reason=reason, market=signal.market_id)
            return RiskDecision(approved=False, reason=reason)

        # 0b. Balance floor check — prevent ruin
        if (
            self._balance_floor_pct > 0
            and self._initial_balance > 0
            and balance < self._initial_balance * self._balance_floor_pct
        ):
            floor = self._initial_balance * self._balance_floor_pct
            reason = (
                f"balance {balance} below floor "
                f"{self._balance_floor_pct * 100}% of initial ({floor})"
            )
            log.warning("risk_rejected", reason=reason, market=signal.market_id)
            return RiskDecision(approved=False, reason=reason)

        # 1. Kill switch check (highest priority)
        if self._kill_switch.is_active:
            reason = "kill switch is active — all trading halted"
            log.warning("risk_rejected", reason=reason, market=signal.market_id)
            return RiskDecision(approved=False, reason=reason)

        # 2. Stop-loss check
        if not self.has_stop_loss(signal):
            reason = "signal missing stop_loss — no naked exposure allowed"
            log.warning("risk_rejected", reason=reason, market=signal.market_id)
            return RiskDecision(approved=False, reason=reason)

        # 3. Position size check (includes estimated fees)
        max_size = balance * self._max_position_pct
        total_cost = position_size + estimated_fee
        if total_cost > max_size:
            reason = (
                f"position size + fees {total_cost} exceeds max "
                f"{self._max_position_pct * 100}% of balance ({max_size})"
            )
            log.warning("risk_rejected", reason=reason, market=signal.market_id)
            return RiskDecision(approved=False, reason=reason, max_size=max_size)

        # 4. Drawdown check (allow trading AT the limit, halt only when exceeded)
        if current_drawdown > self._max_daily_drawdown_pct:
            reason = (
                f"daily drawdown {current_drawdown} > limit "
                f"{self._max_daily_drawdown_pct}"
            )
            log.warning("risk_rejected", reason=reason, market=signal.market_id)
            return RiskDecision(approved=False, reason=reason)

        # 5. Order book impact check
        if book_depth > 0:
            impact = position_size / book_depth
            if impact > self._max_order_book_pct:
                reason = (
                    f"order book impact {impact:.4f} exceeds max "
                    f"{self._max_order_book_pct}"
                )
                log.warning("risk_rejected", reason=reason, market=signal.market_id)
                return RiskDecision(approved=False, reason=reason)

        # All checks passed
        log.info(
            "risk_approved",
            market=signal.market_id,
            position_size=str(position_size),
            drawdown=str(current_drawdown),
        )
        return RiskDecision(approved=True, reason="all checks passed", max_size=position_size)
