"""Kill switch — circuit breaker for daily drawdown protection."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from src.core.logging import get_logger

log = get_logger(__name__)


class KillSwitch:
    """Circuit breaker that halts all trading when drawdown limit is hit.

    State is persisted to Redis when available, falls back to in-memory dict.

    Supports both sync and async Redis clients:
    - For async usage (bot), pass an async Redis client and call
      ``await ks.async_load_state()`` after construction.
    - For sync/no-Redis usage (backtests, tests), pass None or a sync client.
    """

    REDIS_KEY = "mvhe:kill_switch"

    CONSECUTIVE_LOSS_KEY = "mvhe:kill_switch:consecutive_losses"

    def __init__(
        self,
        max_daily_drawdown_pct: Decimal = Decimal("0.05"),
        redis_client: Any | None = None,
        *,
        async_redis: bool = False,
        max_consecutive_losses: int = 0,
    ) -> None:
        self._max_daily_drawdown_pct = max_daily_drawdown_pct
        self._redis = redis_client
        self._async_redis = async_redis
        self._max_consecutive_losses = max_consecutive_losses
        self._consecutive_losses = 0
        self._state: dict[str, Any] = {
            "active": False,
            "reason": "",
            "triggered_at": None,
        }
        # Sync load for backwards compatibility (no-op when async_redis=True
        # or redis_client is None).
        self._sync_load_state()

    # ------------------------------------------------------------------
    # Sync persistence (used when redis_client is a sync redis.Redis or None)
    # ------------------------------------------------------------------

    def _sync_load_state(self) -> None:
        """Load persisted state from a sync Redis client if available."""
        if self._redis is None or self._async_redis:
            return
        try:
            data = self._redis.get(self.REDIS_KEY)
            if data is not None:
                self._state = json.loads(data)
                log.info("kill_switch_state_loaded", active=self._state["active"])
        except Exception:
            log.warning("kill_switch_redis_load_failed", fallback="in-memory")

    def _sync_save_state(self) -> None:
        """Persist state to a sync Redis client if available."""
        if self._redis is None or self._async_redis:
            return
        try:
            self._redis.set(self.REDIS_KEY, json.dumps(self._state))
        except Exception:
            log.warning("kill_switch_redis_save_failed")

    # ------------------------------------------------------------------
    # Async persistence (used when redis_client is redis.asyncio.Redis)
    # ------------------------------------------------------------------

    async def async_load_state(self) -> None:
        """Load persisted state from an async Redis client.

        Call this after construction when using an async Redis client.
        """
        if self._redis is None:
            return
        try:
            data = await self._redis.get(self.REDIS_KEY)
            if data is not None:
                raw = data if isinstance(data, str) else data.decode()
                self._state = json.loads(raw)
                log.info("kill_switch_state_loaded", active=self._state["active"])
        except Exception:
            log.warning("kill_switch_redis_load_failed", fallback="in-memory")

    async def _async_save_state(self) -> None:
        """Persist state to an async Redis client."""
        if self._redis is None:
            return
        try:
            await self._redis.set(self.REDIS_KEY, json.dumps(self._state))
        except Exception:
            log.warning("kill_switch_redis_save_failed")

    # ------------------------------------------------------------------
    # Internal dispatch: pick sync or async save
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        """Persist state via sync Redis.

        Raises RuntimeError if called with an async Redis client,
        since sync save is a no-op in that case and state would be lost.
        Callers with async Redis must use ``async_trigger()`` instead.
        """
        if self._async_redis and self._redis is not None:
            raise RuntimeError(
                "kill_switch.trigger() called with async_redis=True — "
                "state will not persist. Use async_trigger() instead."
            )
        self._sync_save_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """Whether the kill switch is currently active."""
        return bool(self._state["active"])

    def check(self, daily_pnl: Decimal, balance: Decimal) -> bool:
        """Check if drawdown has hit the kill switch threshold.

        Args:
            daily_pnl: Current daily P&L (negative = loss).
            balance: Current account balance.

        Returns:
            True if the kill switch was triggered (or already active).
        """
        if self.is_active:
            return True

        if balance <= 0:
            return False

        drawdown = abs(daily_pnl) / balance if daily_pnl < 0 else Decimal("0")

        if drawdown >= self._max_daily_drawdown_pct:
            reason = (
                f"daily drawdown {drawdown:.4f} >= limit "
                f"{self._max_daily_drawdown_pct}"
            )
            self.trigger(reason)
            return True

        return False

    async def async_check(self, daily_pnl: Decimal, balance: Decimal) -> bool:
        """Async version of check — persists via async Redis on trigger.

        Args:
            daily_pnl: Current daily P&L (negative = loss).
            balance: Current account balance.

        Returns:
            True if the kill switch was triggered (or already active).
        """
        if self.is_active:
            return True

        if balance <= 0:
            return False

        drawdown = abs(daily_pnl) / balance if daily_pnl < 0 else Decimal("0")

        if drawdown >= self._max_daily_drawdown_pct:
            reason = (
                f"daily drawdown {drawdown:.4f} >= limit "
                f"{self._max_daily_drawdown_pct}"
            )
            await self.async_trigger(reason)
            return True

        return False

    def trigger(self, reason: str) -> None:
        """Manually trigger the kill switch.

        Args:
            reason: Human-readable reason for triggering.
        """
        self._state = {
            "active": True,
            "reason": reason,
            "triggered_at": datetime.now(tz=UTC).isoformat(),
        }
        self._save_state()
        log.critical(
            "kill_switch_triggered",
            reason=reason,
            triggered_at=self._state["triggered_at"],
        )

    async def async_trigger(self, reason: str) -> None:
        """Async version of trigger — persists via async Redis."""
        self._state = {
            "active": True,
            "reason": reason,
            "triggered_at": datetime.now(tz=UTC).isoformat(),
        }
        await self._async_save_state()
        log.critical(
            "kill_switch_triggered",
            reason=reason,
            triggered_at=self._state["triggered_at"],
        )

    def reset(self) -> None:
        """Reset the kill switch. Requires explicit call — never automatic."""
        previous_reason = self._state.get("reason", "")
        self._state = {
            "active": False,
            "reason": "",
            "triggered_at": None,
        }
        self._save_state()
        log.warning(
            "kill_switch_reset",
            previous_reason=previous_reason,
        )

    async def async_reset(self) -> None:
        """Async version of reset — persists via async Redis."""
        previous_reason = self._state.get("reason", "")
        self._state = {
            "active": False,
            "reason": "",
            "triggered_at": None,
        }
        await self._async_save_state()
        log.warning(
            "kill_switch_reset",
            previous_reason=previous_reason,
        )

    # ------------------------------------------------------------------
    # Consecutive loss tracking
    # ------------------------------------------------------------------

    def record_trade_result(self, *, is_win: bool) -> bool:
        """Record a trade result and check consecutive loss limit.

        Args:
            is_win: True if the trade was a winner, False if a loser.

        Returns:
            True if the kill switch was triggered by consecutive losses.
        """
        if is_win:
            self._consecutive_losses = 0
            self._sync_save_consecutive_losses()
            return False

        self._consecutive_losses += 1
        self._sync_save_consecutive_losses()

        if (
            self._max_consecutive_losses > 0
            and self._consecutive_losses >= self._max_consecutive_losses
        ):
            reason = (
                f"consecutive losses {self._consecutive_losses} >= limit "
                f"{self._max_consecutive_losses}"
            )
            self.trigger(reason)
            return True
        return False

    async def async_record_trade_result(self, *, is_win: bool) -> bool:
        """Async version of record_trade_result — persists via async Redis.

        Args:
            is_win: True if the trade was a winner, False if a loser.

        Returns:
            True if the kill switch was triggered by consecutive losses.
        """
        if is_win:
            self._consecutive_losses = 0
            await self._async_save_consecutive_losses()
            return False

        self._consecutive_losses += 1
        await self._async_save_consecutive_losses()

        if (
            self._max_consecutive_losses > 0
            and self._consecutive_losses >= self._max_consecutive_losses
        ):
            reason = (
                f"consecutive losses {self._consecutive_losses} >= limit "
                f"{self._max_consecutive_losses}"
            )
            await self.async_trigger(reason)
            return True
        return False

    @property
    def consecutive_losses(self) -> int:
        """Current consecutive loss count."""
        return self._consecutive_losses

    def _sync_save_consecutive_losses(self) -> None:
        if self._redis is None or self._async_redis:
            return
        try:
            self._redis.set(self.CONSECUTIVE_LOSS_KEY, str(self._consecutive_losses))
        except Exception:
            log.warning("kill_switch_consecutive_loss_save_failed")

    async def _async_save_consecutive_losses(self) -> None:
        if self._redis is None:
            return
        try:
            await self._redis.set(
                self.CONSECUTIVE_LOSS_KEY, str(self._consecutive_losses),
            )
        except Exception:
            log.warning("kill_switch_consecutive_loss_save_failed")

    async def async_load_consecutive_losses(self) -> None:
        """Load consecutive loss count from async Redis."""
        if self._redis is None:
            return
        try:
            data = await self._redis.get(self.CONSECUTIVE_LOSS_KEY)
            if data is not None:
                raw = data if isinstance(data, str) else data.decode()
                self._consecutive_losses = int(raw)
                log.info(
                    "kill_switch_consecutive_losses_loaded",
                    count=self._consecutive_losses,
                )
        except Exception:
            log.warning("kill_switch_consecutive_loss_load_failed")
