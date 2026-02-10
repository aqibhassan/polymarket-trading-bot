"""Redis state cache implementing the StateCache protocol."""

from __future__ import annotations

import json
import os
from typing import Any

import redis.asyncio as redis

from src.core.logging import get_logger
from src.models.market import MarketState

log = get_logger(__name__)

DEFAULT_TTL = 300
KEY_PREFIX = "mvhe:"


class RedisCache:
    """Async Redis cache with JSON serialization and key prefixing.

    Implements the StateCache protocol from src.interfaces.
    Reads URL from REDIS_URL env var.
    """

    def __init__(
        self,
        url: str | None = None,
        prefix: str = KEY_PREFIX,
        default_ttl: int = DEFAULT_TTL,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 5.0,
        retry_on_timeout: bool = True,
        health_check_interval: int = 30,
    ) -> None:
        self._url = url or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        self._prefix = prefix
        self._default_ttl = default_ttl
        self._socket_timeout = socket_timeout
        self._socket_connect_timeout = socket_connect_timeout
        self._retry_on_timeout = retry_on_timeout
        self._health_check_interval = health_check_interval
        self._redis: redis.Redis | None = None

    def _key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    async def _get_redis(self) -> redis.Redis:
        if self._redis is None:
            self._redis = redis.from_url(  # type: ignore[no-untyped-call]
                self._url,
                decode_responses=True,
                socket_timeout=self._socket_timeout,
                socket_connect_timeout=self._socket_connect_timeout,
                retry_on_timeout=self._retry_on_timeout,
                health_check_interval=self._health_check_interval,
            )
        return self._redis

    async def ping(self) -> bool:
        """Check Redis connectivity. Returns True if healthy."""
        try:
            r = await self._get_redis()
            return bool(await r.ping())
        except Exception:
            return False

    async def get(self, key: str) -> Any:
        """Get a value by key. Returns None if not found."""
        r = await self._get_redis()
        raw = await r.get(self._key(key))
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set a value with optional TTL (seconds)."""
        r = await self._get_redis()
        serialized = json.dumps(value, default=str)
        effective_ttl = ttl if ttl is not None else self._default_ttl
        await r.set(self._key(key), serialized, ex=effective_ttl)

    async def delete(self, key: str) -> None:
        """Delete a key."""
        r = await self._get_redis()
        await r.delete(self._key(key))

    async def push_to_list(
        self, key: str, value: Any, max_length: int = 20, ttl: int | None = None,
    ) -> None:
        """LPUSH a JSON value and LTRIM to max_length, with optional TTL."""
        r = await self._get_redis()
        full_key = self._key(key)
        serialized = json.dumps(value, default=str)
        async with r.pipeline(transaction=True) as pipe:
            pipe.lpush(full_key, serialized)
            pipe.ltrim(full_key, 0, max_length - 1)
            if ttl is not None:
                pipe.expire(full_key, ttl)
            await pipe.execute()

    async def get_list(
        self, key: str, start: int = 0, end: int = -1,
    ) -> list[Any]:
        """LRANGE with JSON parse for each element."""
        r = await self._get_redis()
        raw_items = await r.lrange(self._key(key), start, end)
        result: list[Any] = []
        for item in raw_items:
            try:
                result.append(json.loads(item))
            except (json.JSONDecodeError, TypeError):
                result.append(item)
        return result

    async def get_market_state(self, market_id: str) -> MarketState | None:
        """Get a cached MarketState by market ID."""
        data = await self.get(f"market:{market_id}")
        if data is None:
            return None
        return MarketState(**data)

    async def set_market_state(self, state: MarketState, ttl: int | None = None) -> None:
        """Cache a MarketState."""
        await self.set(
            f"market:{state.market_id}",
            state.model_dump(mode="json"),
            ttl=ttl,
        )

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
