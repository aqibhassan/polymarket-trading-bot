"""Tests for BacktestEngine — event-driven simulation with no lookahead bias."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

import pytest

from src.backtesting.engine import BacktestEngine, BacktestResult
from src.interfaces import RiskDecision
from src.models.market import (
    Candle,
    MarketState,
    OrderBookLevel,
    OrderBookSnapshot,
    Side,
)
from src.models.signal import Signal, SignalType

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_candle(
    ts: datetime,
    open_: Decimal,
    high: Decimal,
    low: Decimal,
    close: Decimal,
    volume: Decimal = Decimal("1000"),
) -> Candle:
    return Candle(
        exchange="backtest",
        symbol="TEST",
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        timestamp=ts,
    )


def _make_market_state(market_id: str = "test_market") -> MarketState:
    return MarketState(
        market_id=market_id,
        yes_price=Decimal("0.5"),
        no_price=Decimal("0.5"),
        time_remaining_seconds=600,
    )


def _make_orderbook(ts: datetime) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        bids=[OrderBookLevel(price=Decimal("99"), size=Decimal("500"))],
        asks=[OrderBookLevel(price=Decimal("101"), size=Decimal("500"))],
        timestamp=ts,
    )


class MockStrategy:
    """Strategy that emits an ENTRY signal at bar_index == 1."""

    def __init__(self, entry_bar: int = 1) -> None:
        self._entry_bar = entry_bar
        self.received_context_lengths: list[int] = []

    def generate_signals(
        self,
        market_state: MarketState,
        orderbook: OrderBookSnapshot,
        context: dict[str, Any],
    ) -> list[Signal]:
        candles = context["candles"]
        self.received_context_lengths.append(len(candles))

        bar_idx = context["bar_index"]
        if bar_idx == self._entry_bar:
            return [
                Signal(
                    strategy_id="test",
                    market_id=market_state.market_id,
                    signal_type=SignalType.ENTRY,
                    direction=Side.YES,
                    entry_price=Decimal("100"),
                    stop_loss=Decimal("90"),
                    take_profit=Decimal("120"),
                ),
            ]
        return []


class MockRiskManager:
    """Risk manager that always approves with configurable size."""

    def __init__(self, approved: bool = True, max_size: Decimal = Decimal("10")) -> None:
        self._approved = approved
        self._max_size = max_size

    def check_order(
        self,
        signal: Signal,
        position_size: Decimal,
        current_drawdown: Decimal,
        **kwargs: Any,
    ) -> RiskDecision:
        return RiskDecision(
            approved=self._approved,
            reason="" if self._approved else "rejected",
            max_size=self._max_size,
        )


class RejectingRiskManager:
    """Risk manager that always rejects."""

    def check_order(
        self,
        signal: Signal,
        position_size: Decimal,
        current_drawdown: Decimal,
        **kwargs: Any,
    ) -> RiskDecision:
        return RiskDecision(approved=False, reason="risk limit exceeded")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def candles() -> list[Candle]:
    from datetime import timedelta

    base = datetime(2024, 1, 1)
    return [
        _make_candle(
            base + timedelta(minutes=i * 15),
            Decimal("100") + Decimal(str(i)),
            Decimal("105") + Decimal(str(i)),
            Decimal("95") + Decimal(str(i)),
            Decimal("102") + Decimal(str(i)),
        )
        for i in range(5)
    ]


@pytest.fixture
def market_states() -> list[MarketState]:
    return [_make_market_state() for _ in range(5)]


@pytest.fixture
def orderbooks() -> list[OrderBookSnapshot]:
    from datetime import timedelta

    base = datetime(2024, 1, 1)
    return [
        _make_orderbook(base + timedelta(minutes=i * 15))
        for i in range(5)
    ]


@pytest.fixture
def engine() -> BacktestEngine:
    return BacktestEngine(initial_balance=Decimal("10000"))


# ---------------------------------------------------------------------------
# No lookahead bias
# ---------------------------------------------------------------------------

class TestNoLookaheadBias:
    def test_context_only_contains_past_candles(
        self,
        engine: BacktestEngine,
        candles: list[Candle],
        market_states: list[MarketState],
        orderbooks: list[OrderBookSnapshot],
    ) -> None:
        """At bar N, strategy should only see candles[0:N+1]."""
        strategy = MockStrategy(entry_bar=999)  # never triggers
        risk = MockRiskManager()

        engine.run(candles, market_states, orderbooks, strategy, risk)

        # Bar 0 should see 1 candle, bar 1 -> 2, ..., bar 4 -> 5
        assert strategy.received_context_lengths == [1, 2, 3, 4, 5]

    def test_signal_at_bar_n_uses_data_up_to_n(
        self,
        engine: BacktestEngine,
        candles: list[Candle],
        market_states: list[MarketState],
        orderbooks: list[OrderBookSnapshot],
    ) -> None:
        """Signal generated at bar 2 should only have access to bars 0, 1, 2."""

        class ContextCheckingStrategy:
            def generate_signals(
                self,
                market_state: MarketState,
                orderbook: OrderBookSnapshot,
                context: dict[str, Any],
            ) -> list[Signal]:
                bar_idx = context["bar_index"]
                available_candles = context["candles"]
                # Verify: number of available candles == bar_idx + 1
                assert len(available_candles) == bar_idx + 1
                # Verify: last available candle timestamp matches current bar
                if bar_idx < len(candles):
                    assert available_candles[-1].timestamp == candles[bar_idx].timestamp
                return []

        engine.run(
            candles, market_states, orderbooks,
            ContextCheckingStrategy(), MockRiskManager(),
        )


# ---------------------------------------------------------------------------
# Full backtest run
# ---------------------------------------------------------------------------

class TestFullBacktestRun:
    def test_basic_profitable_trade(
        self,
        engine: BacktestEngine,
        candles: list[Candle],
        market_states: list[MarketState],
        orderbooks: list[OrderBookSnapshot],
    ) -> None:
        strategy = MockStrategy(entry_bar=1)
        risk = MockRiskManager(max_size=Decimal("10"))

        result = engine.run(candles, market_states, orderbooks, strategy, risk)

        assert isinstance(result, BacktestResult)
        assert len(result.trades) >= 1
        assert len(result.equity_curve) > 0
        assert "total_return_pct" in result.metrics

    def test_empty_data_returns_empty_result(self, engine: BacktestEngine) -> None:
        result = engine.run([], [], [], MockStrategy(), MockRiskManager())
        assert result.trades == []
        assert result.equity_curve == []

    def test_rejected_signals_produce_no_trades(
        self,
        engine: BacktestEngine,
        candles: list[Candle],
        market_states: list[MarketState],
        orderbooks: list[OrderBookSnapshot],
    ) -> None:
        strategy = MockStrategy(entry_bar=1)
        risk = RejectingRiskManager()

        result = engine.run(candles, market_states, orderbooks, strategy, risk)

        # No trades should open since risk rejects everything
        assert len(result.trades) == 0


# ---------------------------------------------------------------------------
# Stop-loss and take-profit
# ---------------------------------------------------------------------------

class TestExitLogic:
    def test_stop_loss_triggered(self) -> None:
        """Position opened at 100 with SL=90 should close when low <= 90."""
        engine = BacktestEngine(initial_balance=Decimal("10000"))

        candles = [
            _make_candle(
                datetime(2024, 1, 1, 0, 0),
                Decimal("100"), Decimal("105"), Decimal("95"), Decimal("102"),
            ),
            _make_candle(
                datetime(2024, 1, 1, 0, 15),
                Decimal("100"), Decimal("105"), Decimal("95"), Decimal("102"),
            ),
            _make_candle(  # low hits SL
                datetime(2024, 1, 1, 0, 30),
                Decimal("95"), Decimal("96"), Decimal("85"), Decimal("88"),
            ),
            _make_candle(
                datetime(2024, 1, 1, 0, 45),
                Decimal("88"), Decimal("92"), Decimal("86"), Decimal("90"),
            ),
        ]
        market_states = [_make_market_state() for _ in range(4)]
        orderbooks = [_make_orderbook(c.timestamp) for c in candles]

        strategy = MockStrategy(entry_bar=1)
        risk = MockRiskManager(max_size=Decimal("10"))

        result = engine.run(candles, market_states, orderbooks, strategy, risk)

        sl_trades = [t for t in result.trades if t.get("exit_reason") == "stop_loss"]
        assert len(sl_trades) >= 1
        assert sl_trades[0]["exit_price"] == Decimal("90")

    def test_take_profit_triggered(self) -> None:
        """Position opened at 100 with TP=120 should close when high >= 120."""
        engine = BacktestEngine(initial_balance=Decimal("10000"))

        candles = [
            _make_candle(
                datetime(2024, 1, 1, 0, 0),
                Decimal("100"), Decimal("105"), Decimal("95"), Decimal("102"),
            ),
            _make_candle(
                datetime(2024, 1, 1, 0, 15),
                Decimal("100"), Decimal("105"), Decimal("95"), Decimal("102"),
            ),
            _make_candle(  # high hits TP
                datetime(2024, 1, 1, 0, 30),
                Decimal("110"), Decimal("125"), Decimal("108"), Decimal("122"),
            ),
            _make_candle(
                datetime(2024, 1, 1, 0, 45),
                Decimal("122"), Decimal("130"), Decimal("120"), Decimal("125"),
            ),
        ]
        market_states = [_make_market_state() for _ in range(4)]
        orderbooks = [_make_orderbook(c.timestamp) for c in candles]

        strategy = MockStrategy(entry_bar=1)
        risk = MockRiskManager(max_size=Decimal("10"))

        result = engine.run(candles, market_states, orderbooks, strategy, risk)

        tp_trades = [t for t in result.trades if t.get("exit_reason") == "take_profit"]
        assert len(tp_trades) >= 1
        assert tp_trades[0]["exit_price"] == Decimal("120")


# ---------------------------------------------------------------------------
# Equity curve and balance
# ---------------------------------------------------------------------------

class TestEquityCurve:
    def test_equity_curve_length(
        self,
        engine: BacktestEngine,
        candles: list[Candle],
        market_states: list[MarketState],
        orderbooks: list[OrderBookSnapshot],
    ) -> None:
        strategy = MockStrategy(entry_bar=999)
        risk = MockRiskManager()

        result = engine.run(candles, market_states, orderbooks, strategy, risk)

        # initial + one per bar + final settle
        assert len(result.equity_curve) == len(candles) + 2

    def test_no_trades_equity_unchanged(
        self,
        engine: BacktestEngine,
        candles: list[Candle],
        market_states: list[MarketState],
        orderbooks: list[OrderBookSnapshot],
    ) -> None:
        strategy = MockStrategy(entry_bar=999)
        risk = MockRiskManager()

        result = engine.run(candles, market_states, orderbooks, strategy, risk)

        assert result.equity_curve[0] == Decimal("10000")
        assert result.equity_curve[-1] == Decimal("10000")


# ---------------------------------------------------------------------------
# Force-close at backtest end
# ---------------------------------------------------------------------------

class TestForceClose:
    def test_open_positions_closed_at_end(self) -> None:
        """Positions still open at end of data should be force-closed."""
        engine = BacktestEngine(initial_balance=Decimal("100000"))

        # Candles that won't trigger SL or TP
        candles = [
            _make_candle(
                datetime(2024, 1, 1, 0, i * 15),
                Decimal("100"), Decimal("105"),
                Decimal("95"), Decimal("102"),
            )
            for i in range(3)
        ]
        market_states = [_make_market_state() for _ in range(3)]
        orderbooks = [_make_orderbook(c.timestamp) for c in candles]

        # Enter on bar 1, SL at 50 (won't hit), TP at 200 (won't hit)
        class WideStopStrategy:
            def generate_signals(
                self, ms: MarketState, ob: OrderBookSnapshot,
                ctx: dict[str, Any],
            ) -> list[Signal]:
                if ctx["bar_index"] == 1:
                    return [Signal(
                        strategy_id="test",
                        market_id="test",
                        signal_type=SignalType.ENTRY,
                        direction=Side.YES,
                        entry_price=Decimal("100"),
                        stop_loss=Decimal("50"),
                        take_profit=Decimal("200"),
                    )]
                return []

        result = engine.run(
            candles, market_states, orderbooks,
            WideStopStrategy(), MockRiskManager(max_size=Decimal("10")),
        )

        end_trades = [t for t in result.trades if t.get("exit_reason") == "backtest_end"]
        assert len(end_trades) == 1


# ---------------------------------------------------------------------------
# Risk manager integration
# ---------------------------------------------------------------------------

class TestRiskIntegration:
    def test_drawdown_passed_to_risk_manager(self) -> None:
        """Verify that current drawdown is computed and passed to risk_manager."""
        engine = BacktestEngine(initial_balance=Decimal("10000"))

        candles = [
            _make_candle(
                datetime(2024, 1, 1, 0, i * 15),
                Decimal("100"), Decimal("105"),
                Decimal("95"), Decimal("102"),
            )
            for i in range(3)
        ]
        market_states = [_make_market_state() for _ in range(3)]
        orderbooks = [_make_orderbook(c.timestamp) for c in candles]

        received_drawdowns: list[Decimal] = []

        class TrackingRiskManager:
            def check_order(
                self, signal: Signal, position_size: Decimal,
                current_drawdown: Decimal, **kwargs: Any,
            ) -> RiskDecision:
                received_drawdowns.append(current_drawdown)
                return RiskDecision(approved=False, reason="tracking only")

        class AlwaysEntryStrategy:
            def generate_signals(
                self, ms: MarketState, ob: OrderBookSnapshot,
                ctx: dict[str, Any],
            ) -> list[Signal]:
                return [Signal(
                    strategy_id="test",
                    market_id="test",
                    signal_type=SignalType.ENTRY,
                    direction=Side.YES,
                    entry_price=Decimal("100"),
                    stop_loss=Decimal("90"),
                    take_profit=Decimal("120"),
                )]

        engine.run(
            candles, market_states, orderbooks,
            AlwaysEntryStrategy(), TrackingRiskManager(),
        )

        # Should have been called 3 times (once per bar)
        assert len(received_drawdowns) == 3
        # Initial drawdown should be 0
        assert received_drawdowns[0] == Decimal("0")
