"""Performance metrics calculator for backtest results."""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Any

from src.core.logging import get_logger

log = get_logger(__name__)

_ANNUALIZATION_FACTOR = math.sqrt(252)
_ZERO = Decimal("0")


class MetricsCalculator:
    """Calculate performance metrics from trades and an equity curve."""

    def __init__(self, risk_free_rate: float = 0.0) -> None:
        self._risk_free_rate = risk_free_rate

    def calculate(
        self,
        equity_curve: list[Decimal],
        trades: list[dict[str, Any]],
    ) -> dict[str, Decimal | float]:
        """Calculate all performance metrics.

        Args:
            equity_curve: Time series of portfolio values.
            trades: List of trade dicts with at minimum 'pnl' key.

        Returns:
            Dict of metric name to value.
        """
        returns = self._compute_returns(equity_curve)
        pnls = [Decimal(str(t["pnl"])) for t in trades]

        total_return = self._total_return(equity_curve)
        sharpe = self._sharpe_ratio(returns)
        sortino = self._sortino_ratio(returns)
        max_dd = self._max_drawdown(equity_curve)
        calmar = self._calmar_ratio(returns, max_dd)
        win_rate = self._win_rate(pnls)
        profit_factor = self._profit_factor(pnls)
        avg_win_loss = self._avg_win_loss_ratio(pnls)

        metrics: dict[str, Decimal | float] = {
            "total_return_pct": total_return,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown_pct": max_dd,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win_loss_ratio": avg_win_loss,
            "total_trades": Decimal(str(len(trades))),
        }

        log.info("metrics_calculated", **{k: str(v) for k, v in metrics.items()})
        return metrics

    @staticmethod
    def _compute_returns(equity_curve: list[Decimal]) -> list[float]:
        """Compute period-over-period returns from equity curve."""
        if len(equity_curve) < 2:
            return []
        returns: list[float] = []
        for i in range(1, len(equity_curve)):
            prev = equity_curve[i - 1]
            if prev == _ZERO:
                returns.append(0.0)
            else:
                returns.append(float((equity_curve[i] - prev) / prev))
        return returns

    def _sharpe_ratio(self, returns: list[float]) -> float:
        """Sharpe ratio = (mean_return - risk_free) / std * sqrt(252)."""
        if len(returns) < 2:
            return 0.0
        mean_r = sum(returns) / len(returns)
        std_r = _std(returns)
        if std_r == 0.0:
            return 0.0
        return (mean_r - self._risk_free_rate) / std_r * _ANNUALIZATION_FACTOR

    def _sortino_ratio(self, returns: list[float]) -> float:
        """Sortino ratio = (mean_return - risk_free) / downside_std * sqrt(252)."""
        if len(returns) < 2:
            return 0.0
        mean_r = sum(returns) / len(returns)
        downside = [r for r in returns if r < 0]
        if not downside:
            return float("inf") if mean_r > 0 else 0.0
        downside_std = _std(downside)
        if downside_std == 0.0:
            return 0.0
        return (mean_r - self._risk_free_rate) / downside_std * _ANNUALIZATION_FACTOR

    @staticmethod
    def _max_drawdown(equity_curve: list[Decimal]) -> Decimal:
        """Maximum peak-to-trough percentage drawdown."""
        if len(equity_curve) < 2:
            return _ZERO
        peak = equity_curve[0]
        max_dd = _ZERO
        for value in equity_curve[1:]:
            if value > peak:
                peak = value
            if peak > _ZERO:
                dd = (peak - value) / peak
                if dd > max_dd:
                    max_dd = dd
        return max_dd

    def _calmar_ratio(self, returns: list[float], max_dd: Decimal) -> float:
        """Calmar ratio = annualized_return / max_drawdown."""
        if max_dd == _ZERO or len(returns) < 2:
            return 0.0
        mean_r = sum(returns) / len(returns)
        annualized = mean_r * 252
        return annualized / float(max_dd)

    @staticmethod
    def _total_return(equity_curve: list[Decimal]) -> Decimal:
        """Total return percentage."""
        if len(equity_curve) < 2 or equity_curve[0] == _ZERO:
            return _ZERO
        return (equity_curve[-1] - equity_curve[0]) / equity_curve[0]

    @staticmethod
    def _win_rate(pnls: list[Decimal]) -> Decimal:
        """Winning trades / total trades."""
        if not pnls:
            return _ZERO
        wins = sum(1 for p in pnls if p > _ZERO)
        return Decimal(str(wins)) / Decimal(str(len(pnls)))

    @staticmethod
    def _profit_factor(pnls: list[Decimal]) -> Decimal:
        """Gross profit / gross loss."""
        gross_profit: Decimal = sum((p for p in pnls if p > _ZERO), _ZERO)
        gross_loss: Decimal = abs(sum((p for p in pnls if p < _ZERO), _ZERO))
        if gross_loss == _ZERO:
            return Decimal("inf") if gross_profit > _ZERO else _ZERO
        return gross_profit / gross_loss

    @staticmethod
    def _avg_win_loss_ratio(pnls: list[Decimal]) -> Decimal:
        """Average win / average loss."""
        wins = [p for p in pnls if p > _ZERO]
        losses = [abs(p) for p in pnls if p < _ZERO]
        if not wins or not losses:
            return _ZERO
        avg_win = sum(wins) / Decimal(str(len(wins)))
        avg_loss = sum(losses) / Decimal(str(len(losses)))
        if avg_loss == _ZERO:
            return _ZERO
        return avg_win / avg_loss


def _std(values: list[float]) -> float:
    """Population standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)
