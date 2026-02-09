"""Tests for Portfolio — position tracking, P&L, drawdown."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from src.models.market import Position, Side
from src.risk.portfolio import Portfolio


def _make_position(
    market_id: str = "market-1",
    entry_price: Decimal = Decimal("0.50"),
    quantity: Decimal = Decimal("100"),
    side: Side = Side.YES,
    stop_loss: Decimal = Decimal("0.45"),
    take_profit: Decimal = Decimal("0.60"),
) -> Position:
    return Position(
        market_id=market_id,
        side=side,
        token_id="tok-1",
        entry_price=entry_price,
        quantity=quantity,
        entry_time=datetime.now(tz=UTC),
        stop_loss=stop_loss,
        take_profit=take_profit,
    )


class TestAddPosition:
    def test_add_single(self) -> None:
        p = Portfolio()
        pos = _make_position()
        p.add_position(pos)
        assert len(p.open_positions) == 1
        assert p.open_positions[0].market_id == "market-1"

    def test_add_multiple(self) -> None:
        p = Portfolio()
        p.add_position(_make_position(market_id="m1"))
        p.add_position(_make_position(market_id="m2"))
        assert len(p.open_positions) == 2

    def test_open_positions_returns_copy(self) -> None:
        p = Portfolio()
        p.add_position(_make_position())
        positions = p.open_positions
        positions.clear()
        assert len(p.open_positions) == 1


class TestClosePosition:
    def test_close_existing(self) -> None:
        p = Portfolio()
        p.add_position(_make_position(market_id="m1"))
        closed = p.close_position(
            market_id="m1",
            exit_price=Decimal("0.55"),
            exit_reason="profit_target",
        )
        assert closed is not None
        assert closed.exit_price == Decimal("0.55")
        assert closed.exit_reason == "profit_target"
        assert len(p.open_positions) == 0
        assert len(p.closed_positions) == 1

    def test_close_nonexistent_returns_none(self) -> None:
        p = Portfolio()
        result = p.close_position(
            market_id="no-such-market",
            exit_price=Decimal("0.50"),
        )
        assert result is None

    def test_close_with_custom_exit_time(self) -> None:
        p = Portfolio()
        p.add_position(_make_position(market_id="m1"))
        exit_time = datetime(2025, 6, 15, 12, 0, 0, tzinfo=UTC)
        closed = p.close_position(
            market_id="m1",
            exit_price=Decimal("0.55"),
            exit_time=exit_time,
        )
        assert closed is not None
        assert closed.exit_time == exit_time

    def test_close_first_of_multiple(self) -> None:
        p = Portfolio()
        p.add_position(_make_position(market_id="m1"))
        p.add_position(_make_position(market_id="m2"))
        p.close_position(market_id="m1", exit_price=Decimal("0.55"))
        assert len(p.open_positions) == 1
        assert p.open_positions[0].market_id == "m2"

    def test_realized_pnl_positive(self) -> None:
        p = Portfolio()
        p.add_position(
            _make_position(entry_price=Decimal("0.50"), quantity=Decimal("100"))
        )
        closed = p.close_position(market_id="market-1", exit_price=Decimal("0.55"))
        assert closed is not None
        pnl = closed.realized_pnl()
        assert pnl == Decimal("5.00")

    def test_realized_pnl_negative(self) -> None:
        p = Portfolio()
        p.add_position(
            _make_position(entry_price=Decimal("0.50"), quantity=Decimal("100"))
        )
        closed = p.close_position(market_id="market-1", exit_price=Decimal("0.45"))
        assert closed is not None
        pnl = closed.realized_pnl()
        assert pnl == Decimal("-5.00")


class TestDailyPnl:
    def test_unrealized_only(self) -> None:
        p = Portfolio()
        p.add_position(
            _make_position(
                market_id="m1",
                entry_price=Decimal("0.50"),
                quantity=Decimal("100"),
            )
        )
        prices = {"m1": Decimal("0.55")}
        pnl = p.daily_pnl(prices)
        assert pnl == Decimal("5.00")

    def test_realized_only(self) -> None:
        p = Portfolio()
        p.add_position(
            _make_position(
                market_id="m1",
                entry_price=Decimal("0.50"),
                quantity=Decimal("100"),
            )
        )
        p.close_position(market_id="m1", exit_price=Decimal("0.55"))
        pnl = p.daily_pnl({})
        assert pnl == Decimal("5.00")

    def test_combined_realized_and_unrealized(self) -> None:
        p = Portfolio()
        p.add_position(
            _make_position(
                market_id="m1",
                entry_price=Decimal("0.50"),
                quantity=Decimal("100"),
            )
        )
        p.add_position(
            _make_position(
                market_id="m2",
                entry_price=Decimal("0.60"),
                quantity=Decimal("50"),
            )
        )
        p.close_position(market_id="m1", exit_price=Decimal("0.55"))
        prices = {"m2": Decimal("0.65")}
        pnl = p.daily_pnl(prices)
        # realized: (0.55 - 0.50) * 100 = 5.00
        # unrealized: (0.65 - 0.60) * 50 = 2.50
        assert pnl == Decimal("7.50")

    def test_missing_price_uses_entry(self) -> None:
        p = Portfolio()
        p.add_position(
            _make_position(
                market_id="m1",
                entry_price=Decimal("0.50"),
                quantity=Decimal("100"),
            )
        )
        pnl = p.daily_pnl({})
        assert pnl == Decimal("0")

    def test_empty_portfolio(self) -> None:
        p = Portfolio()
        assert p.daily_pnl({}) == Decimal("0")


class TestTotalExposure:
    def test_single_position(self) -> None:
        p = Portfolio()
        p.add_position(
            _make_position(
                market_id="m1",
                entry_price=Decimal("0.50"),
                quantity=Decimal("100"),
            )
        )
        exposure = p.total_exposure({"m1": Decimal("0.55")})
        assert exposure == Decimal("55.00")

    def test_multiple_positions(self) -> None:
        p = Portfolio()
        p.add_position(
            _make_position(
                market_id="m1",
                entry_price=Decimal("0.50"),
                quantity=Decimal("100"),
            )
        )
        p.add_position(
            _make_position(
                market_id="m2",
                entry_price=Decimal("0.60"),
                quantity=Decimal("200"),
            )
        )
        exposure = p.total_exposure({"m1": Decimal("0.55"), "m2": Decimal("0.65")})
        # 0.55*100 + 0.65*200 = 55 + 130 = 185
        assert exposure == Decimal("185.00")

    def test_empty_portfolio_exposure(self) -> None:
        p = Portfolio()
        assert p.total_exposure({}) == Decimal("0")


class TestMaxDrawdown:
    def test_initial_drawdown_zero(self) -> None:
        p = Portfolio()
        assert p.max_drawdown == Decimal("0")

    def test_drawdown_tracks_peak(self) -> None:
        p = Portfolio()
        p.add_position(
            _make_position(
                market_id="m1",
                entry_price=Decimal("0.50"),
                quantity=Decimal("100"),
            )
        )
        # First call: equity = +10 -> peak = 10
        p.daily_pnl({"m1": Decimal("0.60")})
        # Second call: equity = -5 -> drawdown from peak
        p.daily_pnl({"m1": Decimal("0.45")})
        assert p.max_drawdown > Decimal("0")

    def test_drawdown_no_regression(self) -> None:
        """Max drawdown should never decrease."""
        p = Portfolio()
        p.add_position(
            _make_position(
                market_id="m1",
                entry_price=Decimal("0.50"),
                quantity=Decimal("100"),
            )
        )
        p.daily_pnl({"m1": Decimal("0.60")})  # peak
        p.daily_pnl({"m1": Decimal("0.45")})  # drawdown
        dd1 = p.max_drawdown
        p.daily_pnl({"m1": Decimal("0.55")})  # recovery
        dd2 = p.max_drawdown
        assert dd2 >= dd1


class TestDailySummary:
    def test_summary_structure(self) -> None:
        p = Portfolio()
        p.add_position(_make_position(market_id="m1"))
        p.add_position(_make_position(market_id="m2"))
        p.close_position(market_id="m1", exit_price=Decimal("0.55"))
        summary = p.daily_summary()
        assert summary["open_positions"] == 1
        assert summary["closed_positions"] == 1
        assert summary["winners"] == 1
        assert summary["losers"] == 0

    def test_summary_with_loss(self) -> None:
        p = Portfolio()
        p.add_position(
            _make_position(
                market_id="m1",
                entry_price=Decimal("0.50"),
                quantity=Decimal("100"),
            )
        )
        p.close_position(market_id="m1", exit_price=Decimal("0.45"))
        summary = p.daily_summary()
        assert summary["winners"] == 0
        assert summary["losers"] == 1
        assert Decimal(summary["total_realized_pnl"]) == Decimal("-5.00")

    def test_empty_summary(self) -> None:
        p = Portfolio()
        summary = p.daily_summary()
        assert summary["open_positions"] == 0
        assert summary["closed_positions"] == 0
        assert summary["winners"] == 0
        assert summary["losers"] == 0


class TestUpdateEquity:
    """Tests for portfolio.update_equity() — balance-based drawdown tracking."""

    def test_update_equity_sets_peak(self) -> None:
        p = Portfolio()
        p.update_equity(Decimal("10000"))
        assert p._peak_equity == Decimal("10000")
        assert p.max_drawdown == Decimal("0")

    def test_update_equity_tracks_drawdown(self) -> None:
        p = Portfolio()
        p.update_equity(Decimal("10000"))  # peak
        p.update_equity(Decimal("9500"))   # 5% drawdown
        expected_dd = (Decimal("10000") - Decimal("9500")) / Decimal("10000")
        assert p.max_drawdown == expected_dd

    def test_update_equity_drawdown_never_decreases(self) -> None:
        """Max drawdown should never decrease after recovery."""
        p = Portfolio()
        p.update_equity(Decimal("10000"))
        p.update_equity(Decimal("9000"))   # 10% drawdown
        dd_after_drop = p.max_drawdown
        p.update_equity(Decimal("9800"))   # recovery
        assert p.max_drawdown == dd_after_drop

    def test_update_equity_new_high_resets_peak(self) -> None:
        """New all-time high should update peak, but not reduce max_drawdown."""
        p = Portfolio()
        p.update_equity(Decimal("10000"))
        p.update_equity(Decimal("9000"))   # 10% dd
        p.update_equity(Decimal("11000"))  # new peak
        assert p._peak_equity == Decimal("11000")
        assert p.max_drawdown == Decimal("0.1")  # still 10%

    def test_update_equity_multiple_drawdowns_keeps_worst(self) -> None:
        p = Portfolio()
        p.update_equity(Decimal("10000"))
        p.update_equity(Decimal("9500"))   # 5% dd
        p.update_equity(Decimal("10500"))  # new peak
        p.update_equity(Decimal("9450"))   # 10% dd from new peak
        expected_dd = (Decimal("10500") - Decimal("9450")) / Decimal("10500")
        assert p.max_drawdown == expected_dd

    def test_update_equity_zero_balance(self) -> None:
        """Zero balance from starting peak should be 100% drawdown."""
        p = Portfolio()
        p.update_equity(Decimal("10000"))
        p.update_equity(Decimal("0"))
        assert p.max_drawdown == Decimal("1")
