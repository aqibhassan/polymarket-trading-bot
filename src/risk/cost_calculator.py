"""Trade cost calculator — true cost including all friction.

Supports both flat-rate fees and Polymarket's dynamic taker fee curve
for 15-minute crypto markets.
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel

from src.core.logging import get_logger

log = get_logger(__name__)

# Polymarket dynamic fee constant (derived from published fee table).
# fee_per_share = POLY_FEE_K * price * (price * (1 - price))^2
POLY_FEE_K = Decimal("0.25")


class TradeCost(BaseModel):
    """Breakdown of total trade cost."""

    spread_cost: Decimal
    fee_cost: Decimal
    slippage_cost: Decimal
    total_cost: Decimal
    min_profitable_move: Decimal

    model_config = {"frozen": True}


class CostCalculator:
    """Calculates the true cost of a trade including all friction.

    Components: spread cost, exchange fees, and slippage estimate.

    For Polymarket 15-min crypto markets, use ``polymarket_fee`` for the
    dynamic taker fee that peaks at 1.56% at 50/50 odds.
    """

    def __init__(
        self,
        default_fee_rate: Decimal = Decimal("0.002"),
        slippage_bps: Decimal = Decimal("5"),
    ) -> None:
        self._default_fee_rate = default_fee_rate
        self._slippage_bps = slippage_bps

    def calculate(
        self,
        entry_price: Decimal,
        size: Decimal,
        spread: Decimal,
        fee_rate: Decimal | None = None,
    ) -> TradeCost:
        """Calculate total trade cost.

        Args:
            entry_price: Expected entry price.
            size: Position size (quantity of tokens).
            spread: Current bid-ask spread.
            fee_rate: Exchange fee rate (default 0.2%).

        Returns:
            TradeCost with breakdown and min_profitable_move.
        """
        rate = fee_rate if fee_rate is not None else self._default_fee_rate
        notional = entry_price * size

        spread_cost = (spread / Decimal("2")) * size
        fee_cost = notional * rate
        slippage_cost = notional * (self._slippage_bps / Decimal("10000"))
        total_cost = spread_cost + fee_cost + slippage_cost

        # min_profitable_move: price must move at least this much per token
        min_profitable_move = total_cost / size if size > 0 else Decimal("0")

        log.info(
            "cost_calculated",
            entry_price=str(entry_price),
            size=str(size),
            spread_cost=str(spread_cost),
            fee_cost=str(fee_cost),
            slippage_cost=str(slippage_cost),
            total_cost=str(total_cost),
            min_profitable_move=str(min_profitable_move),
        )

        return TradeCost(
            spread_cost=spread_cost,
            fee_cost=fee_cost,
            slippage_cost=slippage_cost,
            total_cost=total_cost,
            min_profitable_move=min_profitable_move,
        )

    @staticmethod
    def polymarket_fee(position_size: Decimal, entry_price: Decimal) -> Decimal:
        """Calculate Polymarket dynamic taker fee for 15-min crypto markets.

        The fee curve peaks at 1.56% at 50/50 odds and drops toward zero
        at the extremes.  Maker fees are 0%; settlement fees are 0%.

        Formula (for N dollars at price P):
            fee = N * 0.25 * P^2 * (1 - P)^2

        Args:
            position_size: Dollar amount of the trade.
            entry_price: Contract price (0 < P < 1).

        Returns:
            Fee in USDC.
        """
        if entry_price <= 0 or entry_price >= 1:
            return Decimal("0")
        return position_size * POLY_FEE_K * entry_price ** 2 * (1 - entry_price) ** 2

    def calculate_binary(
        self,
        entry_price: Decimal,
        position_size: Decimal,
        spread: Decimal = Decimal("0"),
    ) -> TradeCost:
        """Calculate total cost for a binary market trade.

        Uses the Polymarket dynamic fee curve instead of a flat rate.

        The ``position_size`` is the number of shares (as passed to the order
        bridge).  The actual USDC notional is ``position_size * entry_price``.

        Args:
            entry_price: Contract price (0 < P < 1).
            position_size: Number of shares (quantity passed to the bridge).
            spread: Current bid-ask spread.

        Returns:
            TradeCost with breakdown.
        """
        num_shares = position_size
        notional = num_shares * entry_price  # actual USDC wagered
        spread_cost = (spread / Decimal("2")) * num_shares
        fee_cost = self.polymarket_fee(notional, entry_price)
        slippage_cost = notional * (self._slippage_bps / Decimal("10000"))
        total_cost = spread_cost + fee_cost + slippage_cost
        min_profitable_move = total_cost / num_shares if num_shares > 0 else Decimal("0")

        log.info(
            "binary_cost_calculated",
            entry_price=str(entry_price),
            position_size=str(position_size),
            fee_cost=str(fee_cost),
            slippage_cost=str(slippage_cost),
            total_cost=str(total_cost),
        )

        return TradeCost(
            spread_cost=spread_cost,
            fee_cost=fee_cost,
            slippage_cost=slippage_cost,
            total_cost=total_cost,
            min_profitable_move=min_profitable_move,
        )
