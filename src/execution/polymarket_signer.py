"""Polymarket CLOB live order signer and submitter.

Uses py-clob-client to sign and submit orders to the Polymarket CLOB API
via the Polygon network (chain_id=137). All credentials are read from
environment variables -- nothing is ever hardcoded.
"""

from __future__ import annotations

import math
import os
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from src.core.logging import get_logger, log_order_event
from src.execution.rate_limiter import TokenBucketRateLimiter
from src.models.order import Order, OrderSide, OrderStatus, OrderType

logger = get_logger(__name__)

_POLYGON_CHAIN_ID = 137
_SIGNATURE_TYPE = 2  # Gnosis Safe proxy (matches user's Polymarket account)
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
    from py_clob_client.clob_types import OrderType as ClobOrderType

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
    from py_clob_client.client import ClobClient

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
    from py_clob_client.clob_types import OrderArgs

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

    def _fetch_best_ask(self, token_id: str) -> float | None:
        """Fetch the best ask price from the CLOB order book.

        Returns the lowest ask price (float, 2dp) or None if unavailable.
        """
        try:
            book = self._clob_client.get_order_book(token_id)
            asks = book.asks if hasattr(book, "asks") else []
            if not asks:
                logger.debug("order_book_no_asks", token_id=token_id[:16])
                return None
            # asks is a list of OrderBookSummary with .price and .size
            best = min(asks, key=lambda a: float(a.price))
            best_price = float(best.price)
            logger.debug(
                "order_book_best_ask",
                token_id=token_id[:16],
                best_ask=best_price,
                ask_size=float(best.size),
            )
            return best_price
        except Exception as exc:
            logger.debug("order_book_fetch_failed", error=str(exc)[:100])
            return None

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

            # Polymarket CLOB precision constraints:
            #   - price: max 2 decimals (tick = 0.01)
            #   - taker amount (size): max 4 decimals
            #   - maker amount (price * size): max 2 decimals
            # Size rounded to integer: price(2dp) * size(0dp) <= 2dp always.
            #
            # GTC BUY: use the model price directly — the bot already applies
            # confidence-based discounting so the price is intentionally below
            # market to rest as a maker order.
            # Non-GTC BUY: fetch best ask to cross the spread (taker fill).
            # SELL: floor price so ask <= market.
            is_gtc = order_type == OrderType.GTC
            if side == OrderSide.BUY:
                if is_gtc:
                    # GTC: use the model price (already discounted by bot).
                    # Round to 2dp tick and rest as a maker order.
                    rounded_price = math.floor(float(price) * 100) / 100
                    logger.info(
                        "buy_price_gtc_limit",
                        limit_price=rounded_price,
                        model_price=float(price),
                    )
                else:
                    best_ask = self._fetch_best_ask(token_id)
                    if best_ask is not None and best_ask > 0:
                        # Bid at best ask to cross spread (taker fill)
                        rounded_price = best_ask
                        logger.info(
                            "buy_price_from_book",
                            best_ask=best_ask,
                            model_price=float(price),
                        )
                    else:
                        # Fallback: model price + buffer
                        rounded_price = math.ceil(float(price) * 100) / 100 + 0.10
                        logger.info(
                            "buy_price_fallback",
                            rounded_price=rounded_price,
                            model_price=float(price),
                        )
                # Safety cap: honour caller's max_price, fall back to 0.95
                safety_cap = float(max_price) if max_price is not None else 0.95
                if rounded_price > safety_cap:
                    logger.info(
                        "buy_price_capped",
                        original=rounded_price,
                        capped_to=safety_cap,
                        max_price=float(max_price) if max_price else None,
                    )
                rounded_price = min(rounded_price, safety_cap)
                rounded_price = max(rounded_price, 0.01)  # Floor
            else:
                rounded_price = math.floor(float(price) * 100) / 100
                rounded_price = max(rounded_price, 0.01)  # Polymarket min
            # Keep cost within original budget: size * original_price.
            original_cost = float(price) * float(size)
            max_by_cost = math.floor(original_cost / rounded_price) if rounded_price > 0 else 0
            rounded_size = min(math.floor(float(size)), max_by_cost)  # integer tokens

            order_args = _create_order_args(
                token_id=token_id,
                price=rounded_price,
                size=rounded_size,
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
                    "updated_at": datetime.now(tz=UTC),
                    "metadata": {"reject_reason": str(error_msg)},
                })
                self._orders[oid] = order
                log_order_event("live_rejected", oid, reason=str(error_msg))
                return order

            # FOK orders are filled entirely or cancelled — if we get an
            # orderID back, the order was filled. GTC orders sit in the book.
            is_fok = order_type in (OrderType.FOK,)
            fill_status = OrderStatus.FILLED if is_fok else OrderStatus.SUBMITTED
            # Use actual submitted price/size (after book adjustment), not
            # the model price/size the bot originally requested.
            actual_fill_price = Decimal(str(rounded_price)) if is_fok else None
            actual_fill_size = Decimal(str(rounded_size)) if is_fok else None
            order = order.model_copy(update={
                "status": fill_status,
                "exchange_order_id": str(exchange_id),
                "filled_size": actual_fill_size,
                "avg_fill_price": actual_fill_price,
                "updated_at": datetime.now(tz=UTC),
            })
            self._orders[oid] = order
            event_name = "live_filled" if is_fok else "live_submitted"
            log_order_event(
                event_name, oid,
                exchange_order_id=str(exchange_id),
                fill_price=str(rounded_price) if is_fok else None,
                fill_size=str(rounded_size) if is_fok else None,
            )

        except Exception as exc:
            order = order.model_copy(update={
                "status": OrderStatus.REJECTED,
                "updated_at": datetime.now(tz=UTC),
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
                    "updated_at": datetime.now(tz=UTC),
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
        """Fetch current open orders from the CLOB API."""
        await self._rate_limiter.wait()
        try:
            orders: list[dict[str, Any]] = self._clob_client.get_orders()
            logger.info("live_positions_fetched", count=len(orders))
            return orders
        except Exception as exc:
            logger.error("live_positions_fetch_failed", error=str(exc))
            return []

    async def get_balance(self) -> Decimal:
        """Fetch USDC balance from the CLOB API."""
        from py_clob_client.clob_types import AssetType, BalanceAllowanceParams

        await self._rate_limiter.wait()
        try:
            resp = self._clob_client.get_balance_allowance(
                BalanceAllowanceParams(asset_type=AssetType.COLLATERAL),
            )
            # Response is a dict with "balance" key (raw USDC with 6 decimals)
            balance_str = resp.get("balance", "0") if isinstance(resp, dict) else str(resp)
            balance_raw = Decimal(balance_str)
            # Convert from 6-decimal raw to human-readable USDC
            balance = balance_raw / Decimal("1000000")
            logger.info("live_balance_fetched", balance=str(balance))
            return balance
        except Exception as exc:
            logger.error("live_balance_fetch_failed", error=str(exc)[:200])
            return Decimal("0")
