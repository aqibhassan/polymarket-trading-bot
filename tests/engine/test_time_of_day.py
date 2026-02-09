"""Tests for TimeOfDayAnalyzer."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.engine.time_of_day import TimeOfDayAnalyzer, _DEFAULT_HOUR_STATS


@pytest.fixture()
def mock_config() -> MagicMock:
    """ConfigLoader that returns defaults."""
    config = MagicMock()
    config.get.return_value = None  # Use defaults
    return config


class TestTimeOfDayAnalyzer:
    """Tests for TimeOfDayAnalyzer."""

    def test_peak_hour_returns_highest_multiplier(self, mock_config: MagicMock) -> None:
        analyzer = TimeOfDayAnalyzer(mock_config)
        adj = analyzer.get_adjustment(14)
        assert adj.position_size_multiplier == 1.25
        assert adj.estimated_win_rate == 0.93

    def test_off_peak_hour_returns_lowest_multiplier(self, mock_config: MagicMock) -> None:
        analyzer = TimeOfDayAnalyzer(mock_config)
        adj = analyzer.get_adjustment(1)
        assert adj.position_size_multiplier == 0.68
        assert adj.estimated_win_rate == 0.78

    def test_optimal_window_hours_12_to_17(self, mock_config: MagicMock) -> None:
        analyzer = TimeOfDayAnalyzer(mock_config)
        for hour in range(12, 18):
            assert analyzer.is_optimal_window(hour) is True, f"Hour {hour} should be optimal"

    def test_non_optimal_window_hours_0_to_8(self, mock_config: MagicMock) -> None:
        analyzer = TimeOfDayAnalyzer(mock_config)
        for hour in range(0, 9):
            assert analyzer.is_optimal_window(hour) is False, f"Hour {hour} should not be optimal"

    def test_edge_case_hour_0(self, mock_config: MagicMock) -> None:
        analyzer = TimeOfDayAnalyzer(mock_config)
        adj = analyzer.get_adjustment(0)
        assert adj.hour == 0
        assert adj.position_size_multiplier == 0.70
        assert adj.min_confidence_adjustment == 0.03

    def test_edge_case_hour_23(self, mock_config: MagicMock) -> None:
        analyzer = TimeOfDayAnalyzer(mock_config)
        adj = analyzer.get_adjustment(23)
        assert adj.hour == 23
        assert adj.position_size_multiplier == 0.72
        assert adj.min_confidence_adjustment == 0.02

    def test_all_24_hours_return_valid_results(self, mock_config: MagicMock) -> None:
        analyzer = TimeOfDayAnalyzer(mock_config)
        for hour in range(24):
            adj = analyzer.get_adjustment(hour)
            assert adj.hour == hour
            assert 0.0 < adj.estimated_win_rate <= 1.0
            assert adj.position_size_multiplier > 0.0

    def test_multiplier_always_positive(self, mock_config: MagicMock) -> None:
        analyzer = TimeOfDayAnalyzer(mock_config)
        for hour in range(24):
            adj = analyzer.get_adjustment(hour)
            assert adj.position_size_multiplier > 0.0

    def test_out_of_range_hour_clamped_high(self, mock_config: MagicMock) -> None:
        analyzer = TimeOfDayAnalyzer(mock_config)
        adj = analyzer.get_adjustment(25)
        assert adj.hour == 23

    def test_out_of_range_hour_clamped_low(self, mock_config: MagicMock) -> None:
        analyzer = TimeOfDayAnalyzer(mock_config)
        adj = analyzer.get_adjustment(-1)
        assert adj.hour == 0

    def test_config_override_custom_stats(self) -> None:
        custom_stats = {
            i: {"win_rate": 0.50, "size_mult": 0.50, "conf_adj": 0.10}
            for i in range(24)
        }
        config = MagicMock()
        config.get.return_value = custom_stats

        analyzer = TimeOfDayAnalyzer(config)
        adj = analyzer.get_adjustment(14)
        assert adj.position_size_multiplier == 0.50
        assert adj.estimated_win_rate == 0.50
        assert adj.min_confidence_adjustment == 0.10

    def test_adjustment_is_frozen(self, mock_config: MagicMock) -> None:
        analyzer = TimeOfDayAnalyzer(mock_config)
        adj = analyzer.get_adjustment(14)
        with pytest.raises(Exception):  # noqa: B017
            adj.hour = 5  # type: ignore[misc]

    def test_peak_has_negative_confidence_adjustment(self, mock_config: MagicMock) -> None:
        analyzer = TimeOfDayAnalyzer(mock_config)
        adj = analyzer.get_adjustment(14)
        assert adj.min_confidence_adjustment < 0.0

    def test_off_peak_has_positive_confidence_adjustment(self, mock_config: MagicMock) -> None:
        analyzer = TimeOfDayAnalyzer(mock_config)
        adj = analyzer.get_adjustment(0)
        assert adj.min_confidence_adjustment > 0.0
