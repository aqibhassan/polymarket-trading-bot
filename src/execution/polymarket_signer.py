"""Polymarket CLOB live order signer and submitter.

Uses py-clob-client to sign and submit orders to the Polymarket CLOB API
via the Polygon network (chain_id=137). All credentials are read from
environment variables -- nothing is ever hardcoded.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from src.core.logging import get_logger, log_order_event
from src.execution.rate_limiter import TokenBucketRateLimiter
from src.models.order import Order, OrderSide, OrderStatus, OrderType

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

_POLYGON_CHAIN_ID = 137
_SIGNATURE_TYPE = 1
# Rate limit: 10 requests per second with safety margin applied by TokenBucketRateLimiter
_DEFAULT_MAX_TOKENS = 10
_DEFAULT_REFILL_RATE = 10.0


def _mask_secret(value: str) -> str:
    """Mask all but the last 4 characters of a secret."""
    if len(value) <= 4:
        return "****"
    return "*" * (len(value) - 4) + value[-4:]


def _clob_order_type(order_type: OrderType) -> Any:
    """Map internal OrderType to py-clob-client OrderType.

    Import is deferred to avoid module-level import failures when
    py-clob-client's dependency chain is incomplete (dev/test).
    """
    from py_clob_client.clob_types import OrderType as ClobOrderType  # noqa: PLC0415

    if order_type == OrderType.GTC:
        return ClobOrderType.GTC
    if order_type == OrderType.FOK:
        return ClobOrderType.FOK
    # Default to GTC for LIMIT/MARKET
    return ClobOrderType.GTC


def _create_clob_client(
    host: str,
    key: str,
    chain_id: int,
    funder: str | None,
    signature_type: int,
) -> Any:
    """Create a ClobClient instance with deferred import."""
    from py_clob_client.client import ClobClient  # noqa: PLC0415

    return ClobClient(
        host=host,
        key=key,
        chain_id=chain_id,
        funder=funder,
        signature_type=signature_type,
    )


def _create_order_args(
    token_id: str,
    price: float,
    size: float,
    side: str,
) -> Any:
    """Create OrderArgs with deferred import."""
    from py_clob_client.clob_types import OrderArgs  # noqa: PLC0415

    return OrderArgs(
        token_id=token_id,
        price=price,
        size=size,
        side=side,
    )


class PolymarketLiveTrader:
    """Live order signer and submitter for the Polymarket CLOB.

    Implements the ExecutionEngine protocol so it can be used as a
    drop-in replacement for PaperTrader via ExecutionBridge.

    Credentials (env vars):
        POLYMARKET_PRIVATE_KEY  -- Ethereum private key for signing
        POLYMARKET_FUNDER_ADDRESS -- Funder/proxy wallet address
    """

    def __init__(
        self,
        clob_url: str = "https://clob.polymarket.com",
        chain_id: int = _POLYGON_CHAIN_ID,
        rate_limit_tokens: int = _DEFAULT_MAX_TOKENS,
        rate_limit_refill: float = _DEFAULT_REFILL_RATE,
        *,
        _clob_client: Any | None = None,
    ) -> None:
        private_key = os.environ.get("POLYMARKET_PRIVATE_KEY", "")
        funder_address = os.environ.get("POLYMARKET_FUNDER_ADDRESS", "")

        if not private_key:
            msg = "POLYMARKET_PRIVATE_KEY env var is required for live trading"
            raise ValueError(msg)

        logger.info(
            "polymarket_live_trader.init",
            clob_url=clob_url,
            chain_id=chain_id,
            private_key=_mask_secret(private_key),
            funder_address=_mask_secret(funder_address) if funder_address else "not_set",
        )

        if _clob_client is not None:
            # Allow injection for testing
            self._clob_client = _clob_client
        else:
            self._clob_client = _create_clob_client(
                host=clob_url,
                key=private_key,
                chain_id=chain_id,
                funder=funder_address or None,
                signature_type=_SIGNATURE_TYPE,
            )
            # Derive API credentials from the signing key
            self._clob_client.set_api_creds(
                self._clob_client.create_or_derive_api_creds(),
            )

        self._rate_limiter = TokenBucketRateLimiter(
            max_tokens=rate_limit_tokens,
            refill_rate=rate_limit_refill,
        )
        self._orders: dict[str, Order] = {}

    @property
    def mode(self) -> str:
        return "live"

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
        """Sign and submit an order to the Polymarket CLOB.

        Builds a signed order via py-clob-client, submits it, and tracks
        the result internally.
        """
        order = Order(
            market_id=market_id,
            token_id=token_id,
            side=side,
            order_type=order_type,
            price=price,
            size=size,
            strategy_id=strategy_id,
        )
        oid = str(order.id)
        self._orders[oid] = order
        log_order_event(
            "live_submit", oid,
            market_id=market_id, side=side.value,
            price=str(price), size=str(size),
        )

        await self._rate_limiter.wait()

        try:
            # Map side for py-clob-client: "BUY" or "SELL"
            clob_side = "BUY" if side == OrderSide.BUY else "SELL"

            order_args = _create_order_args(
                token_id=token_id,
                price=float(price),
                size=float(size),
                side=clob_side,
            )

            signed_order = self._clob_client.create_order(order_args)
            response: dict[str, Any] = self._clob_client.post_order(
                signed_order,
                _clob_order_type(order_type),
            )

            # Validate response
            if not isinstance(response, dict):
                msg = f"Unexpected response type: {type(response)}"
                raise ValueError(msg)

            exchange_id = response.get("orderID", response.get("id", ""))

            if not exchange_id:
                error_msg = response.get("errorMsg", response.get("error", "unknown"))
                logger.error(
                    "live_order_rejected_by_exchange",
                    order_id=oid,
                    response=str(response),
                )
                order = order.model_copy(update={
                    "status": OrderStatus.REJECTED,
                    "updated_at": datetime.now(tz=timezone.utc),
                    "metadata": {"reject_reason": str(error_msg)},
                })
                self._orders[oid] = order
                log_order_event("live_rejected", oid, reason=str(error_msg))
                return order

            order = order.model_copy(update={
                "status": OrderStatus.SUBMITTED,
                "exchange_order_id": str(exchange_id),
                "updated_at": datetime.now(tz=timezone.utc),
            })
            self._orders[oid] = order
            log_order_event(
                "live_submitted", oid,
                exchange_order_id=str(exchange_id),
            )

        except Exception as exc:
            order = order.model_copy(update={
                "status": OrderStatus.REJECTED,
                "updated_at": datetime.now(tz=timezone.utc),
            })
            self._orders[oid] = order
            log_order_event("live_rejected", oid, reason=str(exc))
            logger.error("live_order_failed", order_id=oid, error=str(exc))

        return order

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order on the Polymarket CLOB."""
        order = self._orders.get(order_id)
        if order is None or order.status.is_terminal:
            return False

        log_order_event("live_cancel_request", order_id)
        await self._rate_limiter.wait()

        try:
            exchange_oid = order.exchange_order_id or order_id
            response = self._clob_client.cancel(exchange_oid)

            # Treat any non-error response as success
            cancelled = bool(response) and not (
                isinstance(response, dict) and response.get("error")
            )

            if cancelled:
                order = order.model_copy(update={
                    "status": OrderStatus.CANCELLED,
                    "updated_at": datetime.now(tz=timezone.utc),
                })
                self._orders[order_id] = order
                log_order_event("live_cancelled", order_id)
            return cancelled

        except Exception as exc:
            logger.error("live_cancel_failed", order_id=order_id, error=str(exc))
            return False

    async def get_order(self, order_id: str) -> Order | None:
        """Retrieve a tracked order by internal ID."""
        return self._orders.get(order_id)

    async def get_positions(self) -> list[dict[str, Any]]:
        """Fetch current open positions from the CLOB API."""
        await self._rate_limiter.wait()
        try:
            positions: list[dict[str, Any]] = self._clob_client.get_positions()
            logger.info("live_positions_fetched", count=len(positions))
            return positions
        except Exception as exc:
            logger.error("live_positions_fetch_failed", error=str(exc))
            return []

    async def get_balance(self) -> Decimal:
        """Fetch USDC balance from the CLOB API."""
        await self._rate_limiter.wait()
        try:
            balance_resp = self._clob_client.get_balance()
            balance = Decimal(str(balance_resp))
            logger.info("live_balance_fetched", balance=str(balance))
            return balance
        except Exception as exc:
            logger.error("live_balance_fetch_failed", error=str(exc))
            return Decimal("0")
