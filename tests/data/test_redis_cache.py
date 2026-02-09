"""Tests for RedisCache."""

from __future__ import annotations

import json
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest

from src.data.redis_cache import DEFAULT_TTL, KEY_PREFIX, RedisCache
from src.models.market import MarketState


@pytest.fixture()
def mock_redis() -> AsyncMock:
    r = AsyncMock()
    r.get = AsyncMock(return_value=None)
    r.set = AsyncMock()
    r.delete = AsyncMock()
    r.aclose = AsyncMock()
    return r


@pytest.fixture()
def cache(mock_redis: AsyncMock) -> RedisCache:
    c = RedisCache(url="redis://localhost:6379/0")
    c._redis = mock_redis
    return c


class TestRedisCacheInit:
    def test_default_url(self) -> None:
        cache = RedisCache()
        assert cache._url == "redis://localhost:6379/0"

    def test_custom_url(self) -> None:
        cache = RedisCache(url="redis://custom:6380/1")
        assert cache._url == "redis://custom:6380/1"

    def test_env_var_url(self) -> None:
        with patch.dict("os.environ", {"REDIS_URL": "redis://env:6379/2"}):
            cache = RedisCache()
            assert cache._url == "redis://env:6379/2"

    def test_prefix(self) -> None:
        cache = RedisCache(prefix="test:")
        assert cache._prefix == "test:"

    def test_default_ttl(self) -> None:
        cache = RedisCache()
        assert cache._default_ttl == DEFAULT_TTL


class TestRedisCacheKeyPrefix:
    def test_key_prefix(self) -> None:
        cache = RedisCache(prefix="mvhe:")
        assert cache._key("foo") == "mvhe:foo"
        assert cache._key("market:123") == "mvhe:market:123"


class TestRedisCacheGet:
    @pytest.mark.asyncio()
    async def test_get_returns_none_when_missing(
        self, cache: RedisCache, mock_redis: AsyncMock
    ) -> None:
        mock_redis.get.return_value = None
        result = await cache.get("missing-key")
        assert result is None
        mock_redis.get.assert_called_once_with(f"{KEY_PREFIX}missing-key")

    @pytest.mark.asyncio()
    async def test_get_returns_deserialized_json(
        self, cache: RedisCache, mock_redis: AsyncMock
    ) -> None:
        mock_redis.get.return_value = json.dumps({"price": "0.55", "volume": 100})
        result = await cache.get("some-key")
        assert result == {"price": "0.55", "volume": 100}

    @pytest.mark.asyncio()
    async def test_get_returns_raw_on_json_error(
        self, cache: RedisCache, mock_redis: AsyncMock
    ) -> None:
        mock_redis.get.return_value = "not-json"
        result = await cache.get("bad-key")
        assert result == "not-json"


class TestRedisCacheSet:
    @pytest.mark.asyncio()
    async def test_set_with_default_ttl(
        self, cache: RedisCache, mock_redis: AsyncMock
    ) -> None:
        await cache.set("key1", {"value": 42})
        mock_redis.set.assert_called_once()
        call_kwargs = mock_redis.set.call_args
        assert call_kwargs[1]["ex"] == DEFAULT_TTL

    @pytest.mark.asyncio()
    async def test_set_with_custom_ttl(
        self, cache: RedisCache, mock_redis: AsyncMock
    ) -> None:
        await cache.set("key1", "hello", ttl=60)
        call_kwargs = mock_redis.set.call_args
        assert call_kwargs[1]["ex"] == 60

    @pytest.mark.asyncio()
    async def test_set_serializes_to_json(
        self, cache: RedisCache, mock_redis: AsyncMock
    ) -> None:
        await cache.set("key1", {"a": 1, "b": "two"})
        call_args = mock_redis.set.call_args[0]
        stored = json.loads(call_args[1])
        assert stored == {"a": 1, "b": "two"}


class TestRedisCacheDelete:
    @pytest.mark.asyncio()
    async def test_delete(self, cache: RedisCache, mock_redis: AsyncMock) -> None:
        await cache.delete("key1")
        mock_redis.delete.assert_called_once_with(f"{KEY_PREFIX}key1")


class TestRedisCacheMarketState:
    @pytest.mark.asyncio()
    async def test_get_market_state_returns_none(
        self, cache: RedisCache, mock_redis: AsyncMock
    ) -> None:
        mock_redis.get.return_value = None
        result = await cache.get_market_state("mkt-123")
        assert result is None

    @pytest.mark.asyncio()
    async def test_set_and_get_market_state(
        self, cache: RedisCache, mock_redis: AsyncMock
    ) -> None:
        state = MarketState(
            market_id="mkt-123",
            yes_price=Decimal("0.62"),
            no_price=Decimal("0.38"),
            time_remaining_seconds=600,
        )

        await cache.set_market_state(state)
        mock_redis.set.assert_called_once()

        # Simulate reading it back
        stored_json = mock_redis.set.call_args[0][1]
        mock_redis.get.return_value = stored_json

        result = await cache.get_market_state("mkt-123")
        assert result is not None
        assert result.market_id == "mkt-123"
        assert result.yes_price == Decimal("0.62")

    @pytest.mark.asyncio()
    async def test_set_market_state_with_ttl(
        self, cache: RedisCache, mock_redis: AsyncMock
    ) -> None:
        state = MarketState(
            market_id="mkt-456",
            yes_price=Decimal("0.50"),
            no_price=Decimal("0.50"),
            time_remaining_seconds=900,
        )
        await cache.set_market_state(state, ttl=120)
        call_kwargs = mock_redis.set.call_args
        assert call_kwargs[1]["ex"] == 120


class TestRedisCacheClose:
    @pytest.mark.asyncio()
    async def test_close(self, cache: RedisCache, mock_redis: AsyncMock) -> None:
        await cache.close()
        mock_redis.aclose.assert_called_once()
        assert cache._redis is None

    @pytest.mark.asyncio()
    async def test_close_when_none(self) -> None:
        c = RedisCache()
        await c.close()
        assert c._redis is None
