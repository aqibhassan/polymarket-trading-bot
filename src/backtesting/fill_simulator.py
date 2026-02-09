"""Realistic fill modeling for backtests."""

from __future__ import annotations

import math
from decimal import Decimal

from pydantic import BaseModel

from src.core.logging import get_logger

log = get_logger(__name__)

_DEFAULT_BASE_SLIPPAGE = Decimal("0.0005")  # 5 bps
_DEFAULT_FEE_RATE = Decimal("0.002")  # 0.2%
_LARGE_ORDER_THRESHOLD = Decimal("0.05")  # 5% of book depth


class SimulatedFill(BaseModel):
    """Result of a simulated fill."""

    fill_price: Decimal
    slippage: Decimal
    fee: Decimal
    net_cost: Decimal

    model_config = {"frozen": True}


class FillSimulator:
    """Simulate realistic order fills with slippage and fees."""

    def __init__(
        self,
        base_slippage: Decimal = _DEFAULT_BASE_SLIPPAGE,
        default_fee_rate: Decimal = _DEFAULT_FEE_RATE,
    ) -> None:
        self._base_slippage = base_slippage
        self._default_fee_rate = default_fee_rate

    def simulate_fill(
        self,
        order_price: Decimal,
        order_size: Decimal,
        book_depth: Decimal,
        fee_rate: Decimal | None = None,
    ) -> SimulatedFill:
        """Simulate a fill with slippage and fees.

        Args:
            order_price: The intended order price.
            order_size: The order size.
            book_depth: Total available depth on the relevant side.
            fee_rate: Fee rate override; defaults to constructor value.

        Returns:
            SimulatedFill with adjusted price, slippage amount, fee, and net cost.
        """
        fee_rate = fee_rate if fee_rate is not None else self._default_fee_rate

        if book_depth <= 0:
            slippage_pct = self._base_slippage
        else:
            size_ratio = order_size / book_depth
            if size_ratio < _LARGE_ORDER_THRESHOLD:
                slippage_pct = size_ratio * self._base_slippage
            else:
                sqrt_ratio = Decimal(str(math.sqrt(float(size_ratio))))
                slippage_pct = sqrt_ratio * self._base_slippage

        slippage_amount = order_price * slippage_pct
        fill_price = order_price + slippage_amount
        notional = fill_price * order_size
        fee = notional * fee_rate
        net_cost = notional + fee

        log.debug(
            "fill_simulated",
            order_price=str(order_price),
            fill_price=str(fill_price),
            slippage=str(slippage_amount),
            fee=str(fee),
        )

        return SimulatedFill(
            fill_price=fill_price,
            slippage=slippage_amount,
            fee=fee,
            net_cost=net_cost,
        )

    def estimate_fee(
        self,
        price: Decimal,
        size: Decimal,
        fee_rate: Decimal | None = None,
    ) -> Decimal:
        """Estimate the fee for a trade without simulating fill/slippage.

        Args:
            price: Expected entry price.
            size: Position size.
            fee_rate: Fee rate override; defaults to constructor value.

        Returns:
            Estimated fee in notional terms.
        """
        rate = fee_rate if fee_rate is not None else self._default_fee_rate
        return price * size * rate
