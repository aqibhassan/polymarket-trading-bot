"""Structured logging foundation for MVHE.

Provides JSON logging (prod) or colored console (dev) via structlog.
Includes an audit trail logger for order/fill/rejection events.
"""

from __future__ import annotations

import os
import sys
from typing import Any, cast

import structlog


def _configure_structlog() -> None:
    """Configure structlog based on MVHE_ENV."""
    env = os.environ.get("MVHE_ENV", "development")
    log_level_name = os.environ.get("MVHE_LOG_LEVEL", "INFO").upper()

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if env == "production":
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Store log level for filtering
    _LOG_LEVELS["current"] = log_level_name


_LOG_LEVELS: dict[str, str] = {"current": "INFO"}
_CONFIGURED = False


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a named logger instance.

    Args:
        name: Logger name (typically module __name__).

    Returns:
        Configured structlog BoundLogger.
    """
    global _CONFIGURED
    if not _CONFIGURED:
        _configure_structlog()
        _CONFIGURED = True

    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))


def get_audit_logger() -> structlog.stdlib.BoundLogger:
    """Get the audit trail logger for orders/fills/rejections.

    All audit events are logged with event_type for downstream filtering.
    """
    return get_logger("mvhe.audit")


def log_order_event(
    action: str,
    order_id: str,
    **kwargs: Any,
) -> None:
    """Log an order lifecycle event to the audit trail.

    Args:
        action: Event type (submit, fill, reject, cancel).
        order_id: Order UUID string.
        **kwargs: Additional context (price, size, reason, etc).
    """
    logger = get_audit_logger()
    logger.info(
        "order_event",
        event_type="audit",
        action=action,
        order_id=order_id,
        **kwargs,
    )
