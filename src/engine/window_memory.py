"""Window Memory — tracks recent window outcomes for serial correlation.

Exploits negative serial correlation at 15-minute BTC timeframes:
after 3+ consecutive same-direction windows, the opposite direction
becomes more likely.
"""

from __future__ import annotations

from collections import deque


class WindowMemory:
    """Track recent window outcomes for serial correlation signals."""

    def __init__(self, max_windows: int = 10) -> None:
        self._outcomes: deque[str] = deque(maxlen=max_windows)

    def record_outcome(self, direction: str) -> None:
        """Record a window outcome (YES = BTC went up, NO = BTC went down)."""
        self._outcomes.append(direction)

    def get_streak(self) -> tuple[str, int]:
        """Get current streak direction and length."""
        if not self._outcomes:
            return ("neutral", 0)
        last = self._outcomes[-1]
        streak = 1
        for i in range(len(self._outcomes) - 2, -1, -1):
            if self._outcomes[i] == last:
                streak += 1
            else:
                break
        return (last, streak)

    def get_contrarian_signal(self) -> tuple[str, float]:
        """Get contrarian signal based on streak length.

        Returns (direction_to_bet, strength).
        After 3+ same-direction windows, bet contrarian.
        """
        direction, streak = self.get_streak()
        if streak < 3:
            return ("neutral", 0.0)
        contrarian = "NO" if direction == "YES" else "YES"
        strength = min((streak - 2) * 0.25, 1.0)
        return (contrarian, strength)
