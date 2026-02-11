"""Auto-claim settled Polymarket positions on-chain.

After a market resolves, winning conditional tokens must be redeemed
to convert them back to USDC. This module handles on-chain redemption
via the ConditionalTokens contract on Polygon.

Uses eth_abi + eth_account + httpx (all already installed via py-clob-client)
to avoid adding web3 as a dependency.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx
from eth_abi import encode
from eth_account import Account

from src.core.logging import get_logger

logger = get_logger(__name__)

# Polygon chain
_CHAIN_ID = 137

# Contract addresses on Polygon
_CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
_NEG_RISK_ADAPTER = "0xC5d563A36AE78145C45a50134d48A1215220f80a"
_USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
_ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"

# Pre-computed function selectors (keccak256 of signature, first 4 bytes)
# balanceOf(address,uint256)
_BALANCE_OF_SEL = bytes.fromhex("00fdd58e")
# redeemPositions(address,bytes32,bytes32,uint256[]) on ConditionalTokens
_CTF_REDEEM_SEL = bytes.fromhex("01b7037c")
# nonce() on Gnosis Safe
_NONCE_SEL = bytes.fromhex("affed0e0")
# getTransactionHash(...) on Gnosis Safe
_GET_TX_HASH_SEL = bytes.fromhex("d8d11f78")
# execTransaction(...) on Gnosis Safe
_EXEC_TX_SEL = bytes.fromhex("6a761202")


def _to_checksum(addr: str) -> str:
    """Convert address to checksummed format."""
    from eth_utils import to_checksum_address
    return to_checksum_address(addr)


def _addr_to_bytes(addr: str) -> bytes:
    """Convert hex address string to 20-byte raw bytes."""
    return bytes.fromhex(addr.replace("0x", "").zfill(40))


def _hex_to_int(hex_str: str) -> int:
    """Convert hex string (with or without 0x) to int."""
    return int(hex_str, 16) if hex_str and hex_str != "0x" else 0


class PolymarketRedeemer:
    """Redeems settled Polymarket conditional tokens on-chain.

    Uses the ConditionalTokens contract on Polygon to redeem winning
    positions back to USDC. Supports both direct EOA transactions and
    Gnosis Safe proxy execution.
    """

    def __init__(
        self,
        rpc_url: str | None = None,
    ) -> None:
        url = rpc_url or os.environ.get(
            "POLYGON_RPC_URL", "https://polygon-rpc.com",
        )
        self._rpc_url = url

        pk = os.environ.get("POLYMARKET_PRIVATE_KEY", "")
        if not pk:
            msg = "POLYMARKET_PRIVATE_KEY required for redemption"
            raise ValueError(msg)

        self._account = Account.from_key(pk)
        self._funder = os.environ.get("POLYMARKET_FUNDER_ADDRESS", "")

        # The address that holds the conditional tokens
        self._token_holder = (
            _to_checksum(self._funder) if self._funder
            else self._account.address
        )

        logger.info(
            "redeemer.init",
            rpc_url=url[:40],
            eoa=self._account.address[:10] + "...",
            token_holder=self._token_holder[:10] + "...",
            use_safe=bool(self._funder),
        )

    # ------------------------------------------------------------------
    # JSON-RPC helpers
    # ------------------------------------------------------------------

    async def _rpc(self, method: str, params: list[Any]) -> Any:
        """Make a JSON-RPC call to the Polygon node."""
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            resp = await client.post(
                self._rpc_url,
                json={
                    "jsonrpc": "2.0",
                    "method": method,
                    "params": params,
                    "id": 1,
                },
            )
            data = resp.json()
            if "error" in data:
                raise RuntimeError(f"RPC error: {data['error']}")
            return data.get("result")

    async def _eth_call(self, to: str, data: bytes) -> bytes:
        """Execute eth_call and return raw bytes result."""
        hex_result = await self._rpc("eth_call", [
            {"to": _to_checksum(to), "data": "0x" + data.hex()},
            "latest",
        ])
        return bytes.fromhex(hex_result.replace("0x", "")) if hex_result else b""

    async def _get_nonce(self) -> int:
        """Get the EOA's transaction nonce."""
        result = await self._rpc(
            "eth_getTransactionCount",
            [self._account.address, "latest"],
        )
        return _hex_to_int(result)

    async def _get_gas_price(self) -> int:
        """Get current gas price."""
        result = await self._rpc("eth_gasPrice", [])
        return _hex_to_int(result)

    async def _send_raw_tx(self, signed_raw: bytes) -> str:
        """Send a signed raw transaction, return tx hash."""
        result = await self._rpc(
            "eth_sendRawTransaction",
            ["0x" + signed_raw.hex()],
        )
        return result

    async def _wait_for_receipt(self, tx_hash: str, timeout: int = 60) -> dict[str, Any] | None:
        """Poll for transaction receipt."""
        for _ in range(timeout // 2):
            result = await self._rpc("eth_getTransactionReceipt", [tx_hash])
            if result is not None:
                return result
            await asyncio.sleep(2)
        return None

    # ------------------------------------------------------------------
    # Contract interaction helpers
    # ------------------------------------------------------------------

    async def _check_ctf_balance(self, token_id: str) -> int:
        """Check ConditionalTokens ERC1155 balance for a token position."""
        tid = int(token_id, 16) if token_id.startswith("0x") else int(token_id)
        call_data = _BALANCE_OF_SEL + encode(
            ["address", "uint256"],
            [self._token_holder, tid],
        )
        result = await self._eth_call(_CTF_ADDRESS, call_data)
        if len(result) >= 32:
            return int.from_bytes(result[:32], "big")
        return 0

    def _build_ctf_redeem_calldata(self, condition_id: str) -> bytes:
        """Build ConditionalTokens.redeemPositions call data.

        redeemPositions(address collateralToken, bytes32 parentCollectionId,
                        bytes32 conditionId, uint256[] indexSets)

        indexSets [1, 2] redeems both YES (index 0 → bitmask 1) and
        NO (index 1 → bitmask 2) outcomes.
        """
        cid_bytes = bytes.fromhex(condition_id.replace("0x", "").zfill(64))
        parent = b"\x00" * 32  # root collection

        return _CTF_REDEEM_SEL + encode(
            ["address", "bytes32", "bytes32", "uint256[]"],
            [
                _to_checksum(_USDC_ADDRESS),
                parent,
                cid_bytes,
                [1, 2],
            ],
        )

    # ------------------------------------------------------------------
    # Transaction submission
    # ------------------------------------------------------------------

    async def _send_direct_tx(self, to: str, call_data: bytes) -> str:
        """Send a transaction directly from the EOA."""
        nonce = await self._get_nonce()
        gas_price = await self._get_gas_price()
        # Use slightly higher gas price for faster inclusion
        gas_price = int(gas_price * 1.2)

        tx = {
            "to": _to_checksum(to),
            "data": call_data,
            "gas": 500_000,
            "gasPrice": gas_price,
            "nonce": nonce,
            "chainId": _CHAIN_ID,
            "value": 0,
        }
        signed = self._account.sign_transaction(tx)
        raw = signed.raw_transaction if hasattr(signed, "raw_transaction") else signed.rawTransaction
        return await self._send_raw_tx(raw)

    async def _send_via_safe(self, to: str, call_data: bytes) -> str:
        """Execute a transaction through a Gnosis Safe proxy wallet.

        For a 1-of-1 Safe (single owner), we:
        1. Get the Safe nonce
        2. Compute the Safe tx hash
        3. Sign with our EOA (eth_sign style, v += 4)
        4. Call execTransaction on the Safe
        """
        from eth_account.messages import encode_defunct

        safe_addr = _to_checksum(self._funder)
        to_addr = _to_checksum(to)
        zero_addr = _to_checksum(_ZERO_ADDRESS)

        # 1. Get Safe nonce
        nonce_result = await self._eth_call(safe_addr, _NONCE_SEL)
        safe_nonce = int.from_bytes(nonce_result[:32], "big") if nonce_result else 0

        # 2. Compute Safe tx hash via getTransactionHash
        get_hash_data = _GET_TX_HASH_SEL + encode(
            [
                "address", "uint256", "bytes", "uint8",
                "uint256", "uint256", "uint256",
                "address", "address", "uint256",
            ],
            [
                to_addr,      # to
                0,            # value
                call_data,    # data
                0,            # operation (Call)
                0,            # safeTxGas
                0,            # baseGas
                0,            # gasPrice
                zero_addr,    # gasToken
                zero_addr,    # refundReceiver
                safe_nonce,   # _nonce
            ],
        )
        safe_tx_hash_raw = await self._eth_call(safe_addr, get_hash_data)
        safe_tx_hash = safe_tx_hash_raw[:32]

        # 3. Sign with eth_sign style (Safe uses v+4 to indicate eth_sign)
        message = encode_defunct(primitive=safe_tx_hash)
        signed_msg = self._account.sign_message(message)
        v = signed_msg.v + 4  # eth_sign indicator for Gnosis Safe
        signature = (
            signed_msg.r.to_bytes(32, "big")
            + signed_msg.s.to_bytes(32, "big")
            + v.to_bytes(1, "big")
        )

        # 4. Build execTransaction call
        exec_data = _EXEC_TX_SEL + encode(
            [
                "address", "uint256", "bytes", "uint8",
                "uint256", "uint256", "uint256",
                "address", "address", "bytes",
            ],
            [
                to_addr,
                0,
                call_data,
                0,          # operation
                0,          # safeTxGas
                0,          # baseGas
                0,          # gasPrice
                zero_addr,  # gasToken
                zero_addr,  # refundReceiver
                signature,
            ],
        )

        # Send the outer transaction from the EOA
        return await self._send_direct_tx(safe_addr, exec_data)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def redeem_positions(
        self,
        condition_id: str,
        token_id: str,
    ) -> str | None:
        """Redeem settled positions for a resolved market.

        Checks the CTF balance for the given token_id. If tokens exist,
        submits a redeemPositions transaction to convert them back to USDC.

        Args:
            condition_id: The market's condition identifier (hex string).
            token_id: The CTF token ID we hold (YES or NO token).

        Returns:
            Transaction hash on success, None if no tokens or on failure.
        """
        if not condition_id:
            logger.warning("redeem.no_condition_id")
            return None

        try:
            # Check if we actually hold tokens
            balance = await self._check_ctf_balance(token_id)
            logger.info(
                "redeem.balance_check",
                condition_id=condition_id[:16],
                token_id=token_id[:16],
                balance=balance,
            )

            if balance == 0:
                logger.info("redeem.no_tokens", condition_id=condition_id[:16])
                return None

            # Build CTF redeemPositions calldata
            redeem_data = self._build_ctf_redeem_calldata(condition_id)
            target = _CTF_ADDRESS

            # Submit transaction (via Safe or direct)
            if self._funder:
                tx_hash = await self._send_via_safe(target, redeem_data)
            else:
                tx_hash = await self._send_direct_tx(target, redeem_data)

            logger.info(
                "redeem.tx_sent",
                condition_id=condition_id[:16],
                tx_hash=tx_hash,
                balance=balance,
                via_safe=bool(self._funder),
            )

            # Wait for receipt
            receipt = await self._wait_for_receipt(tx_hash, timeout=60)
            if receipt:
                status = _hex_to_int(receipt.get("status", "0x0"))
                if status == 1:
                    logger.info(
                        "redeem.success",
                        condition_id=condition_id[:16],
                        tx_hash=tx_hash,
                    )
                    return tx_hash
                else:
                    logger.error(
                        "redeem.tx_reverted",
                        condition_id=condition_id[:16],
                        tx_hash=tx_hash,
                    )
                    return None
            else:
                logger.warning(
                    "redeem.receipt_timeout",
                    condition_id=condition_id[:16],
                    tx_hash=tx_hash,
                )
                # TX might still confirm later — return hash anyway
                return tx_hash

        except Exception as exc:
            logger.error(
                "redeem.failed",
                condition_id=condition_id[:16],
                error=str(exc)[:200],
            )
            return None

    async def check_matic_balance(self) -> float:
        """Check the EOA's MATIC balance (needed for gas)."""
        try:
            result = await self._rpc(
                "eth_getBalance",
                [self._account.address, "latest"],
            )
            wei = _hex_to_int(result)
            return wei / 1e18
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Retroactive sweep — redeem ALL settled positions
    # ------------------------------------------------------------------

    async def sweep_all_settled(self, lookback_hours: int = 48) -> list[str]:
        """Scan for all unredeemed settled BTC 15m positions and redeem them.

        Queries the Gamma API for recently settled BTC 15-minute markets,
        checks CTF balances for both YES and NO tokens, and redeems any
        positions with balance > 0.

        Args:
            lookback_hours: How many hours back to scan for settled markets.

        Returns:
            List of successful redemption transaction hashes.
        """
        tx_hashes: list[str] = []
        try:
            markets = await self._fetch_settled_btc_markets(lookback_hours)
            logger.info(
                "sweep.markets_found",
                count=len(markets),
                lookback_hours=lookback_hours,
            )

            for market in markets:
                condition_id = market.get("condition_id", "")
                yes_token_id = market.get("yes_token_id", "")
                no_token_id = market.get("no_token_id", "")
                question = market.get("question", "")[:60]

                if not condition_id:
                    continue

                # Check both YES and NO token balances
                for label, token_id in [("YES", yes_token_id), ("NO", no_token_id)]:
                    if not token_id:
                        continue
                    try:
                        balance = await self._check_ctf_balance(token_id)
                        if balance > 0:
                            logger.info(
                                "sweep.found_unredeemed",
                                condition_id=condition_id[:16],
                                token=label,
                                token_id=token_id[:16],
                                balance=balance,
                                question=question,
                            )
                            tx_hash = await self.redeem_positions(
                                condition_id=condition_id,
                                token_id=token_id,
                            )
                            if tx_hash:
                                tx_hashes.append(tx_hash)
                                logger.info(
                                    "sweep.redeemed",
                                    tx_hash=tx_hash,
                                    condition_id=condition_id[:16],
                                    token=label,
                                    balance=balance,
                                )
                                # Small delay between redemptions
                                await asyncio.sleep(3)
                    except Exception as exc:
                        logger.warning(
                            "sweep.token_check_failed",
                            condition_id=condition_id[:16],
                            token=label,
                            error=str(exc)[:100],
                        )

        except Exception as exc:
            logger.error("sweep.failed", error=str(exc)[:200])

        logger.info("sweep.complete", redeemed=len(tx_hashes))
        return tx_hashes

    async def _fetch_settled_btc_markets(
        self, lookback_hours: int = 48,
    ) -> list[dict[str, str]]:
        """Fetch recently settled BTC 15-minute markets from the Gamma API."""
        results: list[dict[str, str]] = []
        gamma_url = "https://gamma-api.polymarket.com"

        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
            # Query for closed BTC markets
            resp = await client.get(
                f"{gamma_url}/markets",
                params={
                    "closed": "true",
                    "limit": 100,
                },
            )
            resp.raise_for_status()
            markets = resp.json()

        for m in markets:
            question = m.get("question", "")
            q_lower = question.lower()

            # Filter for BTC 15-minute candle markets only
            if "btc" not in q_lower and "bitcoin" not in q_lower:
                continue
            if "15" not in q_lower and "minute" not in q_lower:
                # Also accept "Up or Down" pattern
                if "up or down" not in q_lower:
                    continue

            condition_id = m.get("condition_id", "") or m.get("conditionId", "")
            if not condition_id:
                continue

            # Extract token IDs
            tokens = m.get("tokens", [])
            yes_token_id = ""
            no_token_id = ""
            for t in tokens:
                outcome = t.get("outcome", "").upper()
                if outcome == "YES":
                    yes_token_id = t.get("token_id", "")
                elif outcome == "NO":
                    no_token_id = t.get("token_id", "")

            if not yes_token_id and not no_token_id:
                continue

            results.append({
                "condition_id": condition_id,
                "yes_token_id": yes_token_id,
                "no_token_id": no_token_id,
                "question": question,
            })

        return results
