"""Time-of-day seasonality adjustment for position sizing and confidence."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel

from src.core.logging import get_logger

if TYPE_CHECKING:
    from src.config.loader import ConfigLoader

log = get_logger(__name__)


class TimeOfDayAdjustment(BaseModel):
    """Hourly adjustment factors for trading."""

    hour: int  # UTC hour 0-23
    estimated_win_rate: float  # estimated win rate for this hour
    position_size_multiplier: float  # scale position size
    min_confidence_adjustment: float  # adjust min confidence threshold
    model_config = {"frozen": True}


# Default hour statistics from singularity backtest (Feb 2026, 64k trades)
# Singularity has a flat hourly profile (~83-86%) with mild peaks at 07/09 UTC.
# size_mult normalised so average hour ~1.0, best hours ~1.10.
_DEFAULT_HOUR_STATS: dict[int, dict[str, float]] = {
    0:  {"win_rate": 0.83, "size_mult": 0.95, "conf_adj": 0.01},
    1:  {"win_rate": 0.83, "size_mult": 0.95, "conf_adj": 0.01},
    2:  {"win_rate": 0.83, "size_mult": 0.95, "conf_adj": 0.01},
    3:  {"win_rate": 0.83, "size_mult": 0.95, "conf_adj": 0.01},
    4:  {"win_rate": 0.83, "size_mult": 0.95, "conf_adj": 0.01},
    5:  {"win_rate": 0.84, "size_mult": 0.98, "conf_adj": 0.00},
    6:  {"win_rate": 0.84, "size_mult": 0.98, "conf_adj": 0.00},
    7:  {"win_rate": 0.86, "size_mult": 1.10, "conf_adj": -0.01},
    8:  {"win_rate": 0.85, "size_mult": 1.05, "conf_adj": -0.01},
    9:  {"win_rate": 0.86, "size_mult": 1.10, "conf_adj": -0.01},
    10: {"win_rate": 0.85, "size_mult": 1.05, "conf_adj": 0.00},
    11: {"win_rate": 0.85, "size_mult": 1.05, "conf_adj": 0.00},
    12: {"win_rate": 0.84, "size_mult": 1.00, "conf_adj": 0.00},
    13: {"win_rate": 0.84, "size_mult": 1.00, "conf_adj": 0.00},
    14: {"win_rate": 0.84, "size_mult": 1.00, "conf_adj": 0.00},
    15: {"win_rate": 0.84, "size_mult": 1.00, "conf_adj": 0.00},
    16: {"win_rate": 0.84, "size_mult": 1.00, "conf_adj": 0.00},
    17: {"win_rate": 0.84, "size_mult": 0.98, "conf_adj": 0.00},
    18: {"win_rate": 0.84, "size_mult": 0.98, "conf_adj": 0.00},
    19: {"win_rate": 0.83, "size_mult": 0.95, "conf_adj": 0.01},
    20: {"win_rate": 0.83, "size_mult": 0.95, "conf_adj": 0.01},
    21: {"win_rate": 0.83, "size_mult": 0.95, "conf_adj": 0.01},
    22: {"win_rate": 0.83, "size_mult": 0.95, "conf_adj": 0.01},
    23: {"win_rate": 0.83, "size_mult": 0.95, "conf_adj": 0.01},
}


class TimeOfDayAnalyzer:
    """Hourly seasonality adjustment for trading signals."""

    def __init__(self, config: ConfigLoader) -> None:
        custom_stats: dict[int, dict[str, float]] | None = config.get(
            "strategy.singularity.hour_stats"
        )
        if custom_stats is not None:
            self._hour_stats = custom_stats
            log.info("time_of_day_init", source="config_override", hours=len(custom_stats))
        else:
            self._hour_stats = _DEFAULT_HOUR_STATS
            log.info("time_of_day_init", source="defaults", hours=len(self._hour_stats))

    def get_adjustment(self, utc_hour: int) -> TimeOfDayAdjustment:
        """Get trading adjustment for the given UTC hour.

        Args:
            utc_hour: Hour of day in UTC (0-23).

        Returns:
            TimeOfDayAdjustment with size multiplier and confidence adjustment.
        """
        clamped = max(0, min(23, utc_hour))
        stats = self._hour_stats[clamped]
        return TimeOfDayAdjustment(
            hour=clamped,
            estimated_win_rate=stats["win_rate"],
            position_size_multiplier=stats["size_mult"],
            min_confidence_adjustment=stats["conf_adj"],
        )

    def is_optimal_window(self, utc_hour: int) -> bool:
        """Check if current hour is in the optimal trading window."""
        adj = self.get_adjustment(utc_hour)
        return adj.position_size_multiplier >= 1.0

    def get_current_adjustment(self) -> TimeOfDayAdjustment:
        """Get adjustment for the current UTC hour."""
        return self.get_adjustment(datetime.now(tz=UTC).hour)
