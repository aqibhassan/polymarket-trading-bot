"""Execution bridge — routes to paper or live backend."""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING, Any

from src.core.logging import get_logger

if TYPE_CHECKING:
    from decimal import Decimal

    from src.execution.circuit_breaker import CircuitBreaker
    from src.execution.order_manager import OrderManager
    from src.execution.paper_trader import PaperTrader
    from src.execution.polymarket_signer import PolymarketLiveTrader
    from src.models.order import Order, OrderSide, OrderType

logger = get_logger(__name__)

# Polymarket CLOB minimum order size (tokens)
POLYMARKET_MIN_ORDER_SIZE = 5


def _confirm_live_trading() -> bool:
    """Prompt the user to confirm live trading. Returns True if confirmed."""
    print(
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
        circuit_breaker: CircuitBreaker | None = None,
        *,
        skip_confirmation: bool = False,
        order_timeout_seconds: float = 0,
        order_max_retries: int = 0,
    ) -> None:
        self._mode = mode
        self._paper_trader = paper_trader
        self._order_manager = order_manager
        self._live_trader = live_trader
        self._circuit_breaker = circuit_breaker
        self._order_timeout = order_timeout_seconds
        self._order_max_retries = order_max_retries

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
                print("Live trading declined. Exiting.", file=sys.stderr)
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
        max_price: Decimal | None = None,
        expiration: int | None = None,
    ) -> Order:
        # Minimum order size gate (Polymarket CLOB requires >= 5 tokens)
        if self._mode == "live" and size < POLYMARKET_MIN_ORDER_SIZE:
            from src.models.order import Order as OrderModel
            from src.models.order import OrderStatus

            logger.warning(
                "bridge_order_below_minimum",
                market_id=market_id,
                size=str(size),
                minimum=POLYMARKET_MIN_ORDER_SIZE,
            )
            rejected = OrderModel(
                market_id=market_id,
                token_id=token_id,
                side=side,
                order_type=order_type,
                price=price,
                size=size,
                strategy_id=strategy_id,
                status=OrderStatus.REJECTED,
            )
            return rejected

        # Circuit breaker gate
        if self._circuit_breaker is not None and not self._circuit_breaker.can_execute():
            from src.models.order import Order as OrderModel
            from src.models.order import OrderStatus

            logger.warning("bridge_circuit_breaker_open", market_id=market_id)
            rejected = OrderModel(
                market_id=market_id,
                token_id=token_id,
                side=side,
                order_type=order_type,
                price=price,
                size=size,
                strategy_id=strategy_id,
                status=OrderStatus.REJECTED,
            )
            return rejected

        last_error: Exception | None = None
        attempts = 1 + self._order_max_retries

        for attempt in range(attempts):
            try:
                # Build kwargs — only pass max_price/expiration to backends that support it
                _kwargs: dict[str, Any] = {}
                if max_price is not None:
                    _kwargs["max_price"] = max_price
                if expiration is not None:
                    _kwargs["expiration"] = expiration
                if self._order_timeout > 0:
                    order = await asyncio.wait_for(
                        self._backend().submit_order(
                            market_id, token_id, side, order_type, price, size, strategy_id,
                            **_kwargs,
                        ),
                        timeout=self._order_timeout,
                    )
                else:
                    order = await self._backend().submit_order(
                        market_id, token_id, side, order_type, price, size, strategy_id,
                        **_kwargs,
                    )

                if self._circuit_breaker is not None:
                    self._circuit_breaker.record_success()
                return order

            except TimeoutError as exc:
                last_error = exc
                logger.warning(
                    "bridge_order_timeout",
                    attempt=attempt + 1,
                    max_attempts=attempts,
                    timeout=self._order_timeout,
                    market_id=market_id,
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "bridge_order_failed",
                    attempt=attempt + 1,
                    max_attempts=attempts,
                    error=str(exc),
                    market_id=market_id,
                )

        # All retries exhausted — record failure and return rejected order
        if self._circuit_breaker is not None:
            self._circuit_breaker.record_failure()

        from src.models.order import Order as OrderModel
        from src.models.order import OrderStatus

        logger.error(
            "bridge_order_all_retries_exhausted",
            market_id=market_id,
            attempts=attempts,
            last_error=str(last_error),
        )
        rejected = OrderModel(
            market_id=market_id,
            token_id=token_id,
            side=side,
            order_type=order_type,
            price=price,
            size=size,
            strategy_id=strategy_id,
            status=OrderStatus.REJECTED,
        )
        return rejected

    async def cancel_order(self, order_id: str) -> bool:
        return await self._backend().cancel_order(order_id)

    async def get_order(self, order_id: str) -> Order | None:
        return await self._backend().get_order(order_id)

    async def get_order_status(self, exchange_order_id: str) -> dict[str, Any]:
        """Query the CLOB for an order's status by exchange order ID.

        Only meaningful in live mode. Paper mode returns a stub.
        """
        if self._live_trader is not None and hasattr(self._live_trader, "get_order_status"):
            return await self._live_trader.get_order_status(exchange_order_id)
        return {"status": "UNKNOWN"}
