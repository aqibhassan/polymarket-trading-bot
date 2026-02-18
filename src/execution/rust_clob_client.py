"""Polymarket CLOB client — Rust-native (fast) with Python fallback.

Tries to import the native Rust extension module `rs_clob_python` built
via PyO3 + maturin.  When unavailable (dev, CI, non-x86), falls back
transparently to the pure-Python `py-clob-client`.
"""

from __future__ import annotations

import os
from typing import Any

from src.core.logging import get_logger

logger = get_logger(__name__)

try:
    from rs_clob_python import RustClobClient as _RustClient  # type: ignore[import-not-found]

    RUST_AVAILABLE = True
    logger.info("rust_clob_client.available", backend="rust")
except ImportError:
    RUST_AVAILABLE = False
    logger.info("rust_clob_client.unavailable", backend="python_fallback")


def _use_python_fallback() -> bool:
    """Check if the env-var kill-switch is set."""
    return os.environ.get("MVHE_USE_PYTHON_CLOB", "").lower() in ("1", "true", "yes")


class ClobClientWrapper:
    """Unified async interface wrapping either Rust or Python CLOB client."""

    def __init__(
        self,
        host: str,
        key: str,
        chain_id: int,
        funder: str | None,
        signature_type: int,
    ) -> None:
        if RUST_AVAILABLE and not _use_python_fallback():
            self._backend = "rust"
            self._client: Any = _RustClient(host, key, chain_id, funder, signature_type)
            logger.info("clob_client_wrapper.init", backend="rust")
        else:
            self._backend = "python"
            from py_clob_client.client import ClobClient

            self._client = ClobClient(
                host=host,
                key=key,
                chain_id=chain_id,
                funder=funder,
                signature_type=signature_type,
            )
            self._client.set_api_creds(self._client.create_or_derive_api_creds())
            logger.info("clob_client_wrapper.init", backend="python")

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def raw_client(self) -> Any:
        """Escape hatch: access the underlying client for advanced use."""
        return self._client

    # ------------------------------------------------------------------
    # Hot path: create + sign + post in one call
    # ------------------------------------------------------------------
    async def create_and_post_order(
        self,
        token_id: str,
        price: float,
        size: float,
        side: str,
        order_type: str,
        expiration: int = 0,
    ) -> dict[str, Any]:
        """Sign and post an order in one call (Rust) or two calls (Python)."""
        if self._backend == "rust":
            return await self._client.create_and_post_order(
                token_id, str(price), str(size), side, order_type,
                expiration if expiration else None,
            )
        else:
            from py_clob_client.clob_types import OrderArgs, OrderType as ClobOrderType

            args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=side,
                expiration=expiration,
            )
            signed = self._client.create_order(args)

            ot_map = {
                "GTC": ClobOrderType.GTC,
                "FOK": ClobOrderType.FOK,
                "FAK": ClobOrderType.FAK,
                "GTD": ClobOrderType.GTD,
            }
            clob_ot = ot_map.get(order_type.upper(), ClobOrderType.GTC)
            return self._client.post_order(signed, clob_ot)

    # ------------------------------------------------------------------
    # Order book
    # ------------------------------------------------------------------
    async def get_order_book(self, token_id: str) -> Any:
        """Fetch full order book for a token."""
        if self._backend == "rust":
            return await self._client.get_order_book(token_id)
        return self._client.get_order_book(token_id)

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------
    async def cancel(self, order_id: str) -> Any:
        """Cancel an order by exchange order ID."""
        if self._backend == "rust":
            return await self._client.cancel(order_id)
        return self._client.cancel(order_id)

    # ------------------------------------------------------------------
    # Get single order
    # ------------------------------------------------------------------
    async def get_order(self, order_id: str) -> Any:
        """Get order details by exchange order ID."""
        if self._backend == "rust":
            return await self._client.get_order(order_id)
        return self._client.get_order(order_id)

    # ------------------------------------------------------------------
    # Get all orders
    # ------------------------------------------------------------------
    async def get_orders(self) -> Any:
        """Get all orders for the authenticated user."""
        if self._backend == "rust":
            return await self._client.get_orders()
        return self._client.get_orders()

    # ------------------------------------------------------------------
    # Balance
    # ------------------------------------------------------------------
    async def get_balance_allowance(self, params: Any = None) -> Any:
        """Get USDC balance and allowances."""
        if self._backend == "rust":
            return await self._client.get_balance_allowance()
        if params is not None:
            return self._client.get_balance_allowance(params)
        from py_clob_client.clob_types import AssetType, BalanceAllowanceParams
        return self._client.get_balance_allowance(
            BalanceAllowanceParams(asset_type=AssetType.COLLATERAL),
        )

    # ------------------------------------------------------------------
    # Midpoint
    # ------------------------------------------------------------------
    async def get_midpoint(self, token_id: str) -> Any:
        """Get midpoint price for a token."""
        if self._backend == "rust":
            return await self._client.get_midpoint(token_id)
        return self._client.get_midpoint(token_id)
