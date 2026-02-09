"""Audit logger — append-only trail for all trading events."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.core.logging import get_logger, log_order_event

if TYPE_CHECKING:
    from src.data.clickhouse_store import ClickHouseStore
    from src.models.order import Fill, Order
    from src.models.signal import Signal

logger = get_logger(__name__)

_DEFAULT_LOG_PATH = "logs/audit.jsonl"


class AuditLogger:
    """Append-only audit trail for all trading events.

    Always writes to local JSONL file. Optionally also persists
    to ClickHouse when a ``ClickHouseStore`` is provided.
    """

    def __init__(
        self,
        log_path: str = _DEFAULT_LOG_PATH,
        clickhouse_store: ClickHouseStore | None = None,
    ) -> None:
        self._log_path = Path(log_path)
        self._clickhouse_store = clickhouse_store
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    async def log_order(self, order: Order, action: str) -> None:
        entry = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "event_type": "order",
            "action": action,
            "order_id": str(order.id),
            "market_id": order.market_id,
            "side": order.side.value,
            "order_type": order.order_type.value,
            "price": str(order.price),
            "size": str(order.size),
            "status": order.status.value,
            "strategy_id": order.strategy_id,
        }
        log_order_event(action, str(order.id), market_id=order.market_id)
        await self._persist(entry)

    async def log_fill(self, fill: Fill, order: Order) -> None:
        entry = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "event_type": "fill",
            "action": "fill",
            "order_id": str(order.id),
            "fill_id": str(fill.id),
            "price": str(fill.price),
            "size": str(fill.size),
            "fee": str(fill.fee),
        }
        log_order_event("fill", str(order.id), fill_price=str(fill.price), fill_size=str(fill.size))
        await self._persist(entry)

    async def log_rejection(self, signal: Signal, reason: str) -> None:
        entry = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "event_type": "rejection",
            "action": "rejected",
            "strategy_id": signal.strategy_id,
            "market_id": signal.market_id,
            "reason": reason,
        }
        log_order_event("rejected", signal.strategy_id, reason=reason, market_id=signal.market_id)
        await self._persist(entry)

    async def _persist(self, entry: dict[str, Any]) -> None:
        # Always write to local file
        self._write_file(entry)

        # Optionally replicate to ClickHouse
        if self._clickhouse_store is not None:
            try:
                await self._clickhouse_store.insert_audit_event(
                    {
                        "event_id": f"{entry.get('event_type', 'unknown')}_{entry.get('order_id', entry.get('strategy_id', ''))}",
                        "order_id": entry.get("order_id", ""),
                        "event_type": entry.get("event_type", ""),
                        "market_id": entry.get("market_id", ""),
                        "strategy": entry.get("strategy_id", ""),
                        "details": json.dumps(entry),
                        "timestamp": datetime.now(tz=timezone.utc),
                    },
                )
            except Exception as exc:
                logger.warning("clickhouse_audit_write_failed", error=str(exc))

    def _write_file(self, entry: dict[str, Any]) -> None:
        with open(self._log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
