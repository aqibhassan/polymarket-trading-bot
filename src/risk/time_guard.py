"""Time guard — prevents entry too late and forces exit before resolution."""

from __future__ import annotations

from src.core.logging import get_logger

log = get_logger(__name__)


class TimeGuard:
    """Guards against entering too late in a candle and forces timely exits.

    Uses configurable thresholds for entry cutoff and forced exit.
    """

    def __init__(
        self,
        no_entry_after_minute: float = 8.0,
        force_exit_minute: float = 11.0,
        warning_threshold_minutes: float = 3.0,
    ) -> None:
        self._no_entry_after = no_entry_after_minute
        self._force_exit = force_exit_minute
        self._warning_threshold = warning_threshold_minutes

    def can_enter(self, minutes_elapsed: float) -> bool:
        """Check whether a new entry is allowed at the current candle time.

        Args:
            minutes_elapsed: Minutes elapsed since candle open.

        Returns:
            True if entry is allowed.
        """
        allowed = minutes_elapsed <= self._no_entry_after
        if not allowed:
            log.info(
                "entry_blocked",
                minutes_elapsed=minutes_elapsed,
                cutoff=self._no_entry_after,
            )
        return allowed

    def must_exit(self, minutes_elapsed: float) -> bool:
        """Check whether positions must be exited.

        Args:
            minutes_elapsed: Minutes elapsed since candle open.

        Returns:
            True if forced exit is required.
        """
        forced = minutes_elapsed >= self._force_exit
        if forced:
            log.warning(
                "forced_exit",
                minutes_elapsed=minutes_elapsed,
                threshold=self._force_exit,
            )
        return forced

    def time_remaining_warning(self, minutes_elapsed: float) -> str | None:
        """Return a warning string if close to forced exit, else None.

        Args:
            minutes_elapsed: Minutes elapsed since candle open.

        Returns:
            Warning message if < warning_threshold minutes to force exit,
            otherwise None.
        """
        remaining = self._force_exit - minutes_elapsed
        if remaining <= 0:
            return "FORCE EXIT NOW — past force exit time"
        if remaining < self._warning_threshold:
            return f"WARNING: {remaining:.1f} minutes to forced exit"
        return None
