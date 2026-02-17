"""Tests for Window Memory."""

from __future__ import annotations

import pytest

from src.engine.window_memory import WindowMemory


class TestWindowMemory:
    def test_empty_memory(self) -> None:
        """Empty memory → neutral streak."""
        mem = WindowMemory()
        direction, streak = mem.get_streak()
        assert direction == "neutral"
        assert streak == 0

    def test_streak_detection(self) -> None:
        """3 consecutive YES → streak of 3."""
        mem = WindowMemory()
        mem.record_outcome("YES")
        mem.record_outcome("YES")
        mem.record_outcome("YES")
        direction, streak = mem.get_streak()
        assert direction == "YES"
        assert streak == 3

    def test_streak_broken(self) -> None:
        """YES YES NO → streak of 1 (NO)."""
        mem = WindowMemory()
        mem.record_outcome("YES")
        mem.record_outcome("YES")
        mem.record_outcome("NO")
        direction, streak = mem.get_streak()
        assert direction == "NO"
        assert streak == 1

    def test_contrarian_signal_3_streak(self) -> None:
        """3+ YES streak → contrarian NO signal."""
        mem = WindowMemory()
        for _ in range(3):
            mem.record_outcome("YES")
        direction, strength = mem.get_contrarian_signal()
        assert direction == "NO"
        assert strength == pytest.approx(0.25)

    def test_contrarian_signal_5_streak(self) -> None:
        """5 NO streak → contrarian YES with strength 0.75."""
        mem = WindowMemory()
        for _ in range(5):
            mem.record_outcome("NO")
        direction, strength = mem.get_contrarian_signal()
        assert direction == "YES"
        assert strength == pytest.approx(0.75)

    def test_contrarian_signal_capped(self) -> None:
        """6+ streak → strength capped at 1.0."""
        mem = WindowMemory()
        for _ in range(8):
            mem.record_outcome("YES")
        direction, strength = mem.get_contrarian_signal()
        assert direction == "NO"
        assert strength == pytest.approx(1.0)

    def test_no_signal_short_streak(self) -> None:
        """Streak < 3 → neutral."""
        mem = WindowMemory()
        mem.record_outcome("YES")
        mem.record_outcome("YES")
        direction, strength = mem.get_contrarian_signal()
        assert direction == "neutral"
        assert strength == 0.0

    def test_max_windows(self) -> None:
        """Only retains max_windows outcomes."""
        mem = WindowMemory(max_windows=3)
        for _ in range(5):
            mem.record_outcome("YES")
        assert len(mem._outcomes) == 3
