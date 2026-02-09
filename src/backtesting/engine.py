"""Event-driven backtesting engine — no lookahead bias."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from src.backtesting.fill_simulator import FillSimulator
from src.backtesting.metrics import MetricsCalculator
from src.core.logging import get_logger
from src.models.signal import SignalType

if TYPE_CHECKING:
    from src.interfaces import RiskDecision
    from src.models.market import Candle, MarketState, OrderBookSnapshot

log = get_logger(__name__)

_ZERO = Decimal("0")


class BacktestResult(BaseModel):
    """Container for backtest output."""

    trades: list[dict[str, Any]] = Field(default_factory=list)
    equity_curve: list[Decimal] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class BacktestEngine:
    """Event-driven backtesting simulator.

    Iterates through historical candles in time order, feeds data up to
    the current bar only (no lookahead), generates signals, checks risk,
    and simulates fills.
    """

    def __init__(
        self,
        initial_balance: Decimal = Decimal("10000"),
        fill_simulator: FillSimulator | None = None,
        metrics_calculator: MetricsCalculator | None = None,
    ) -> None:
        self._initial_balance = initial_balance
        self._fill_sim = fill_simulator or FillSimulator()
        self._metrics_calc = metrics_calculator or MetricsCalculator()

    def run(
        self,
        candles: list[Candle],
        market_states: list[MarketState],
        orderbooks: list[OrderBookSnapshot],
        strategy: Any,
        risk_manager: Any,
    ) -> BacktestResult:
        """Run a full backtest.

        Args:
            candles: Historical candle data, sorted by time.
            market_states: Per-candle market state snapshots.
            orderbooks: Per-candle orderbook snapshots.
            strategy: Object with generate_signals(market_state, orderbook, context).
            risk_manager: Object with check_order(signal, position_size, drawdown).

        Returns:
            BacktestResult with trades, equity curve, and metrics.
        """
        balance = self._initial_balance
        trades: list[dict[str, Any]] = []
        equity_curve: list[Decimal] = [balance]
        open_positions: list[dict[str, Any]] = []

        n_bars = min(len(candles), len(market_states), len(orderbooks))
        if n_bars == 0:
            log.warning("backtest_empty", reason="no data")
            return BacktestResult()

        log.info("backtest_start", bars=n_bars, initial_balance=str(balance))

        for i in range(n_bars):
            candle = candles[i]
            market_state = market_states[i]
            orderbook = orderbooks[i]

            # Check exits on open positions using current candle
            closed = self._check_exits(open_positions, candle)
            for pos in closed:
                gross_pnl = (pos["exit_price"] - pos["entry_price"]) * pos["quantity"]
                # Subtract entry fee from PnL (exit fee already excluded from balance)
                pnl = gross_pnl - pos["entry_fee"]
                balance += pos["quantity"] * pos["exit_price"]
                pos["pnl"] = pnl
                trades.append(pos)
                open_positions.remove(pos)

            # Build context: only candles up to and including current bar (no lookahead)
            context: dict[str, Any] = {
                "candles": candles[: i + 1],
                "bar_index": i,
            }

            # Generate signals from strategy
            signals = strategy.generate_signals(market_state, orderbook, context)

            for signal in signals:
                if signal.signal_type != SignalType.ENTRY:
                    continue

                # Risk check (include fee estimate for accurate sizing)
                current_dd = self._current_drawdown(equity_curve)
                entry_price = signal.entry_price or candle.close
                # Proposed new position cost (notional), not cumulative
                proposed_size = entry_price * Decimal("1")
                estimated_fee = self._fill_sim.estimate_fee(entry_price, proposed_size)
                decision: RiskDecision = risk_manager.check_order(
                    signal=signal,
                    position_size=proposed_size,
                    current_drawdown=current_dd,
                    balance=balance,
                    estimated_fee=estimated_fee,
                )
                if not decision.approved:
                    log.debug("signal_rejected", reason=decision.reason)
                    continue

                entry_price = signal.entry_price or candle.close
                size = decision.max_size if decision.max_size > _ZERO else Decimal("1")

                # Check sufficient balance
                cost = entry_price * size
                if cost > balance:
                    log.debug("insufficient_balance", cost=str(cost), balance=str(balance))
                    continue

                # Simulate fill
                book_depth = orderbook.total_bid_depth + orderbook.total_ask_depth
                sim_fill = self._fill_sim.simulate_fill(
                    entry_price, size, book_depth,
                )

                balance -= sim_fill.net_cost

                position = {
                    "market_id": signal.market_id,
                    "side": signal.direction.value,
                    "entry_price": sim_fill.fill_price,
                    "quantity": size,
                    "entry_time": candle.timestamp,
                    "stop_loss": signal.stop_loss or _ZERO,
                    "take_profit": signal.take_profit or _ZERO,
                    "entry_fee": sim_fill.fee,
                    "slippage": sim_fill.slippage,
                }
                open_positions.append(position)
                log.debug(
                    "position_opened",
                    market=signal.market_id,
                    price=str(sim_fill.fill_price),
                    size=str(size),
                )

            # Compute mark-to-market equity: cash + current market value of positions
            position_market_value = sum(
                candle.close * p["quantity"]
                for p in open_positions
            )
            equity_curve.append(balance + position_market_value)

        # Force-close any remaining open positions at last candle close
        if open_positions and n_bars > 0:
            last_candle = candles[n_bars - 1]
            for pos in list(open_positions):
                pos["exit_price"] = last_candle.close
                pos["exit_time"] = last_candle.timestamp
                pos["exit_reason"] = "backtest_end"
                gross_pnl = (pos["exit_price"] - pos["entry_price"]) * pos["quantity"]
                pnl = gross_pnl - pos["entry_fee"]
                balance += pos["quantity"] * pos["exit_price"]
                pos["pnl"] = pnl
                trades.append(pos)
            open_positions.clear()

        equity_curve.append(balance)

        metrics = self._metrics_calc.calculate(equity_curve, trades)
        log.info("backtest_complete", total_trades=len(trades))

        return BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            metrics=metrics,
        )

    @staticmethod
    def _check_exits(
        open_positions: list[dict[str, Any]],
        candle: Candle,
    ) -> list[dict[str, Any]]:
        """Check stop-loss and take-profit against current candle."""
        to_close: list[dict[str, Any]] = []
        for pos in open_positions:
            sl = pos["stop_loss"]
            tp = pos["take_profit"]

            # Stop-loss hit
            if sl > _ZERO and candle.low <= sl:
                pos["exit_price"] = sl
                pos["exit_time"] = candle.timestamp
                pos["exit_reason"] = "stop_loss"
                to_close.append(pos)
                continue

            # Take-profit hit
            if tp > _ZERO and candle.high >= tp:
                pos["exit_price"] = tp
                pos["exit_time"] = candle.timestamp
                pos["exit_reason"] = "take_profit"
                to_close.append(pos)

        return to_close

    @staticmethod
    def _current_drawdown(equity_curve: list[Decimal]) -> Decimal:
        """Compute current drawdown from peak."""
        if not equity_curve:
            return _ZERO
        peak = max(equity_curve)
        current = equity_curve[-1]
        if peak == _ZERO:
            return _ZERO
        return (peak - current) / peak
