"""Tests for BacktestReporter — JSON and console output."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path  # noqa: TCH003

import pytest

from src.backtesting.reporter import BacktestReporter


@pytest.fixture
def reporter() -> BacktestReporter:
    return BacktestReporter()


@pytest.fixture
def sample_result() -> dict:
    return {
        "metrics": {
            "total_return_pct": Decimal("0.15"),
            "sharpe_ratio": 1.85,
            "sortino_ratio": 2.30,
            "max_drawdown_pct": Decimal("0.08"),
            "calmar_ratio": 1.50,
            "win_rate": Decimal("0.65"),
            "profit_factor": Decimal("2.1"),
            "avg_win_loss_ratio": Decimal("1.8"),
            "total_trades": Decimal("50"),
        },
        "trades": [
            {"pnl": str(Decimal("10")), "market_id": "test"},
            {"pnl": str(Decimal("-5")), "market_id": "test"},
        ],
        "equity_curve": [str(Decimal("1000")), str(Decimal("1150"))],
        "validation": {
            "sharpe_ci_95": (0.8, 2.5),
            "p_value": 0.02,
            "is_significant": True,
            "overfitting_warning": None,
        },
    }


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

class TestJsonOutput:
    def test_valid_json(self, reporter: BacktestReporter, sample_result: dict) -> None:
        json_str = reporter.generate_json(sample_result)
        parsed = json.loads(json_str)
        assert "metrics" in parsed
        assert "trades" in parsed

    def test_decimal_serialized_as_string(self, reporter: BacktestReporter) -> None:
        result = {"value": Decimal("1.23456789")}
        json_str = reporter.generate_json(result)
        parsed = json.loads(json_str)
        assert parsed["value"] == "1.23456789"

    def test_empty_result(self, reporter: BacktestReporter) -> None:
        json_str = reporter.generate_json({})
        parsed = json.loads(json_str)
        assert parsed == {}


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

class TestConsoleSummary:
    def test_summary_contains_metrics(
        self, reporter: BacktestReporter, sample_result: dict,
    ) -> None:
        summary = reporter.print_summary(sample_result)
        assert "BACKTEST SUMMARY" in summary
        assert "Total Return" in summary
        assert "Sharpe Ratio" in summary
        assert "Max Drawdown" in summary
        assert "Win Rate" in summary

    def test_summary_shows_percentages(
        self, reporter: BacktestReporter, sample_result: dict,
    ) -> None:
        summary = reporter.print_summary(sample_result)
        assert "15.00%" in summary  # total return 0.15 = 15%

    def test_summary_shows_validation(
        self, reporter: BacktestReporter, sample_result: dict,
    ) -> None:
        summary = reporter.print_summary(sample_result)
        assert "Validation" in summary
        assert "Sharpe 95% CI" in summary
        assert "P-value" in summary
        assert "Significant" in summary

    def test_summary_without_validation(self, reporter: BacktestReporter) -> None:
        result: dict = {"metrics": {"total_return_pct": Decimal("0.1")}}
        summary = reporter.print_summary(result)
        assert "Validation" not in summary

    def test_summary_with_overfitting_warning(self, reporter: BacktestReporter) -> None:
        result: dict = {
            "metrics": {},
            "validation": {
                "sharpe_ci_95": (0.5, 1.5),
                "p_value": 0.04,
                "is_significant": True,
                "overfitting_warning": "IS/OOS ratio 3.5x",
            },
        }
        summary = reporter.print_summary(result)
        assert "WARNING" in summary
        assert "3.5x" in summary


# ---------------------------------------------------------------------------
# File saving
# ---------------------------------------------------------------------------

class TestFileSaving:
    def test_save_creates_file(
        self, reporter: BacktestReporter, sample_result: dict, tmp_path: Path,
    ) -> None:
        out_path = tmp_path / "reports" / "test_report.json"
        reporter.save_report(sample_result, out_path)
        assert out_path.exists()

        content = json.loads(out_path.read_text())
        assert "metrics" in content

    def test_save_creates_parent_dirs(
        self, reporter: BacktestReporter, tmp_path: Path,
    ) -> None:
        out_path = tmp_path / "a" / "b" / "c" / "report.json"
        reporter.save_report({"test": True}, out_path)
        assert out_path.exists()
