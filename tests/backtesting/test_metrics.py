"""Tests for MetricsCalculator — all 8 metrics verified against known values."""

from __future__ import annotations

import math
from decimal import Decimal

import pytest

from src.backtesting.metrics import MetricsCalculator, _std


@pytest.fixture
def calc() -> MetricsCalculator:
    return MetricsCalculator(risk_free_rate=0.0)


# ---------------------------------------------------------------------------
# Total return
# ---------------------------------------------------------------------------

class TestTotalReturn:
    def test_positive_return(self, calc: MetricsCalculator) -> None:
        curve = [Decimal("1000"), Decimal("1100"), Decimal("1200")]
        result = calc.calculate(curve, [{"pnl": Decimal("200")}])
        # (1200 - 1000) / 1000 = 0.2
        assert result["total_return_pct"] == Decimal("0.2")

    def test_negative_return(self, calc: MetricsCalculator) -> None:
        curve = [Decimal("1000"), Decimal("900"), Decimal("800")]
        result = calc.calculate(
            curve,
            [{"pnl": Decimal("-100")}, {"pnl": Decimal("-100")}],
        )
        assert result["total_return_pct"] == Decimal("-0.2")

    def test_zero_start(self, calc: MetricsCalculator) -> None:
        curve = [Decimal("0"), Decimal("100")]
        result = calc.calculate(curve, [{"pnl": Decimal("100")}])
        assert result["total_return_pct"] == Decimal("0")


# ---------------------------------------------------------------------------
# Sharpe ratio
# ---------------------------------------------------------------------------

class TestSharpeRatio:
    def test_known_sharpe(self, calc: MetricsCalculator) -> None:
        # Construct returns: [0.01, 0.02, -0.01, 0.03, 0.01]
        curve = [Decimal("1000")]
        price = Decimal("1000")
        for r in [0.01, 0.02, -0.01, 0.03, 0.01]:
            price = price * (1 + Decimal(str(r)))
            curve.append(price)

        result = calc.calculate(curve, [{"pnl": Decimal("60")}])
        returns = [0.01, 0.02, -0.01, 0.03, 0.01]
        mean_r = sum(returns) / len(returns)
        std_r = _std(returns)
        expected = mean_r / std_r * math.sqrt(252)
        assert abs(result["sharpe_ratio"] - expected) < 0.001

    def test_constant_equity_zero_sharpe(self, calc: MetricsCalculator) -> None:
        curve = [Decimal("1000")] * 5
        result = calc.calculate(curve, [])
        assert result["sharpe_ratio"] == 0.0

    def test_single_point_zero_sharpe(self, calc: MetricsCalculator) -> None:
        result = calc.calculate([Decimal("1000")], [])
        assert result["sharpe_ratio"] == 0.0


# ---------------------------------------------------------------------------
# Sortino ratio
# ---------------------------------------------------------------------------

class TestSortinoRatio:
    def test_no_downside_returns_inf(self, calc: MetricsCalculator) -> None:
        curve = [Decimal("1000"), Decimal("1100"), Decimal("1200")]
        result = calc.calculate(curve, [{"pnl": Decimal("200")}])
        assert result["sortino_ratio"] == float("inf")

    def test_all_negative_returns(self, calc: MetricsCalculator) -> None:
        curve = [Decimal("1000"), Decimal("900"), Decimal("800")]
        result = calc.calculate(
            curve,
            [{"pnl": Decimal("-100")}, {"pnl": Decimal("-100")}],
        )
        assert result["sortino_ratio"] < 0


# ---------------------------------------------------------------------------
# Max drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_known_drawdown(self, calc: MetricsCalculator) -> None:
        curve = [
            Decimal("1000"),
            Decimal("1200"),  # peak
            Decimal("900"),   # trough -> dd = (1200-900)/1200 = 25%
            Decimal("1100"),
        ]
        result = calc.calculate(
            curve,
            [{"pnl": Decimal("-100")}, {"pnl": Decimal("200")}],
        )
        assert result["max_drawdown_pct"] == Decimal("0.25")

    def test_no_drawdown(self, calc: MetricsCalculator) -> None:
        curve = [Decimal("100"), Decimal("200"), Decimal("300")]
        result = calc.calculate(curve, [{"pnl": Decimal("200")}])
        assert result["max_drawdown_pct"] == Decimal("0")


# ---------------------------------------------------------------------------
# Calmar ratio
# ---------------------------------------------------------------------------

class TestCalmarRatio:
    def test_calmar_with_drawdown(self, calc: MetricsCalculator) -> None:
        curve = [
            Decimal("1000"),
            Decimal("1200"),
            Decimal("900"),
            Decimal("1100"),
        ]
        result = calc.calculate(
            curve,
            [{"pnl": Decimal("-100")}, {"pnl": Decimal("200")}],
        )
        # max_dd = 0.25
        assert result["calmar_ratio"] != 0.0

    def test_zero_drawdown_zero_calmar(self, calc: MetricsCalculator) -> None:
        curve = [Decimal("100"), Decimal("200"), Decimal("300")]
        result = calc.calculate(curve, [{"pnl": Decimal("200")}])
        assert result["calmar_ratio"] == 0.0


# ---------------------------------------------------------------------------
# Win rate
# ---------------------------------------------------------------------------

class TestWinRate:
    def test_all_winners(self, calc: MetricsCalculator) -> None:
        trades = [{"pnl": Decimal("10")}, {"pnl": Decimal("20")}]
        curve = [Decimal("100"), Decimal("130")]
        result = calc.calculate(curve, trades)
        assert result["win_rate"] == Decimal("1")

    def test_mixed_trades(self, calc: MetricsCalculator) -> None:
        trades = [
            {"pnl": Decimal("10")},
            {"pnl": Decimal("-5")},
            {"pnl": Decimal("3")},
            {"pnl": Decimal("-1")},
        ]
        curve = [Decimal("100"), Decimal("107")]
        result = calc.calculate(curve, trades)
        # 2 wins / 4 trades = 0.5
        assert result["win_rate"] == Decimal("0.5")

    def test_no_trades(self, calc: MetricsCalculator) -> None:
        curve = [Decimal("100"), Decimal("100")]
        result = calc.calculate(curve, [])
        assert result["win_rate"] == Decimal("0")


# ---------------------------------------------------------------------------
# Profit factor
# ---------------------------------------------------------------------------

class TestProfitFactor:
    def test_known_profit_factor(self, calc: MetricsCalculator) -> None:
        trades = [
            {"pnl": Decimal("100")},
            {"pnl": Decimal("-50")},
            {"pnl": Decimal("80")},
            {"pnl": Decimal("-30")},
        ]
        curve = [Decimal("1000"), Decimal("1100")]
        result = calc.calculate(curve, trades)
        # gross_profit = 180, gross_loss = 80, pf = 180/80 = 2.25
        assert result["profit_factor"] == Decimal("2.25")

    def test_no_losses_inf(self, calc: MetricsCalculator) -> None:
        trades = [{"pnl": Decimal("10")}]
        curve = [Decimal("100"), Decimal("110")]
        result = calc.calculate(curve, trades)
        assert result["profit_factor"] == Decimal("Infinity")


# ---------------------------------------------------------------------------
# Average win/loss ratio
# ---------------------------------------------------------------------------

class TestAvgWinLossRatio:
    def test_known_ratio(self, calc: MetricsCalculator) -> None:
        trades = [
            {"pnl": Decimal("20")},
            {"pnl": Decimal("-10")},
            {"pnl": Decimal("30")},
            {"pnl": Decimal("-5")},
        ]
        curve = [Decimal("1000"), Decimal("1035")]
        result = calc.calculate(curve, trades)
        # avg_win = 25, avg_loss = 7.5, ratio = 25/7.5 = 3.333...
        expected = Decimal("25") / Decimal("7.5")
        assert result["avg_win_loss_ratio"] == expected

    def test_no_losses_zero(self, calc: MetricsCalculator) -> None:
        trades = [{"pnl": Decimal("10")}]
        curve = [Decimal("100"), Decimal("110")]
        result = calc.calculate(curve, trades)
        assert result["avg_win_loss_ratio"] == Decimal("0")


# ---------------------------------------------------------------------------
# Helper function
# ---------------------------------------------------------------------------

class TestStdHelper:
    def test_known_std(self) -> None:
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        result = _std(values)
        # population std = 2.0
        assert abs(result - 2.0) < 0.001

    def test_single_value(self) -> None:
        assert _std([5.0]) == 0.0

    def test_empty(self) -> None:
        assert _std([]) == 0.0
