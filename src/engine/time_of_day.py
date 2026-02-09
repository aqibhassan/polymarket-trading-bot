"""Time-of-day seasonality adjustment for position sizing and confidence."""

from __future__ import annotations

from datetime import datetime, timezone
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


# Default hour statistics from backtest research
# Key insight: 14:00-16:00 UTC (US market open) has highest accuracy
_DEFAULT_HOUR_STATS: dict[int, dict[str, float]] = {
    0:  {"win_rate": 0.79, "size_mult": 0.70, "conf_adj": 0.03},
    1:  {"win_rate": 0.78, "size_mult": 0.68, "conf_adj": 0.03},
    2:  {"win_rate": 0.78, "size_mult": 0.68, "conf_adj": 0.03},
    3:  {"win_rate": 0.79, "size_mult": 0.70, "conf_adj": 0.03},
    4:  {"win_rate": 0.80, "size_mult": 0.72, "conf_adj": 0.02},
    5:  {"win_rate": 0.80, "size_mult": 0.72, "conf_adj": 0.02},
    6:  {"win_rate": 0.81, "size_mult": 0.75, "conf_adj": 0.02},
    7:  {"win_rate": 0.82, "size_mult": 0.78, "conf_adj": 0.02},
    8:  {"win_rate": 0.83, "size_mult": 0.80, "conf_adj": 0.01},
    9:  {"win_rate": 0.84, "size_mult": 0.85, "conf_adj": 0.01},
    10: {"win_rate": 0.86, "size_mult": 0.90, "conf_adj": 0.00},
    11: {"win_rate": 0.87, "size_mult": 0.95, "conf_adj": 0.00},
    12: {"win_rate": 0.89, "size_mult": 1.00, "conf_adj": -0.02},
    13: {"win_rate": 0.91, "size_mult": 1.10, "conf_adj": -0.03},
    14: {"win_rate": 0.93, "size_mult": 1.25, "conf_adj": -0.05},
    15: {"win_rate": 0.92, "size_mult": 1.20, "conf_adj": -0.04},
    16: {"win_rate": 0.88, "size_mult": 1.05, "conf_adj": -0.02},
    17: {"win_rate": 0.87, "size_mult": 1.00, "conf_adj": -0.01},
    18: {"win_rate": 0.86, "size_mult": 0.95, "conf_adj": 0.00},
    19: {"win_rate": 0.85, "size_mult": 0.90, "conf_adj": 0.00},
    20: {"win_rate": 0.84, "size_mult": 0.85, "conf_adj": 0.01},
    21: {"win_rate": 0.83, "size_mult": 0.80, "conf_adj": 0.01},
    22: {"win_rate": 0.81, "size_mult": 0.75, "conf_adj": 0.02},
    23: {"win_rate": 0.80, "size_mult": 0.72, "conf_adj": 0.02},
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
        return self.get_adjustment(datetime.now(tz=timezone.utc).hour)
