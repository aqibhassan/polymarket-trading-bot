"""Execution bridge — routes to paper or live backend."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from src.core.logging import get_logger

if TYPE_CHECKING:
    from decimal import Decimal

    from src.execution.order_manager import OrderManager
    from src.execution.paper_trader import PaperTrader
    from src.execution.polymarket_signer import PolymarketLiveTrader
    from src.models.order import Order, OrderSide, OrderType

logger = get_logger(__name__)


def _confirm_live_trading() -> bool:
    """Prompt the user to confirm live trading. Returns True if confirmed."""
    print(  # noqa: T201
        "\n*** WARNING: You are about to start LIVE trading with REAL funds. ***\n"
        "Type 'YES' to confirm: ",
        end="",
        flush=True,
    )
    try:
        response = input()
        return response.strip() == "YES"
    except (EOFError, KeyboardInterrupt):
        return False


class ExecutionBridge:
    """Routes ExecutionEngine calls to PaperTrader, OrderManager, or PolymarketLiveTrader.

    Implements the ExecutionEngine protocol.
    """

    def __init__(
        self,
        mode: str,
        paper_trader: PaperTrader | None = None,
        order_manager: OrderManager | None = None,
        live_trader: PolymarketLiveTrader | None = None,
        *,
        skip_confirmation: bool = False,
    ) -> None:
        self._mode = mode
        self._paper_trader = paper_trader
        self._order_manager = order_manager
        self._live_trader = live_trader

        if mode == "paper" and paper_trader is None:
            msg = "paper_trader required when mode='paper'"
            raise ValueError(msg)
        if mode == "live" and order_manager is None and live_trader is None:
            msg = "order_manager or live_trader required when mode='live'"
            raise ValueError(msg)

        # CRITICAL SAFETY: require confirmation for live trading
        if mode == "live" and not skip_confirmation:
            if not _confirm_live_trading():
                logger.warning("live_trading_declined")
                print("Live trading declined. Exiting.", file=sys.stderr)  # noqa: T201
                msg = "Live trading requires explicit confirmation"
                raise SystemExit(msg)
            logger.info("live_trading_confirmed")

    @property
    def mode(self) -> str:
        return self._mode

    def _backend(self) -> PaperTrader | OrderManager | PolymarketLiveTrader:
        if self._mode == "paper":
            assert self._paper_trader is not None
            return self._paper_trader
        if self._live_trader is not None:
            return self._live_trader
        assert self._order_manager is not None
        return self._order_manager

    async def submit_order(
        self,
        market_id: str,
        token_id: str,
        side: OrderSide,
        order_type: OrderType,
        price: Decimal,
        size: Decimal,
        strategy_id: str = "",
    ) -> Order:
        return await self._backend().submit_order(
            market_id, token_id, side, order_type, price, size, strategy_id,
        )

    async def cancel_order(self, order_id: str) -> bool:
        return await self._backend().cancel_order(order_id)

    async def get_order(self, order_id: str) -> Order | None:
        return await self._backend().get_order(order_id)
