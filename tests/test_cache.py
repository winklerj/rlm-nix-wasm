"""Tests for the content-addressed cache store."""

import tempfile
from pathlib import Path

import pytest

from rlm.cache.store import CacheStore, make_cache_key
from rlm.evaluator.lightweight import LightweightEvaluator
from rlm.types import OpType, Operation


@pytest.fixture
def cache_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d) / "cache"


@pytest.fixture
def cache(cache_dir):
    return CacheStore(cache_dir)


class TestCacheStore:
    def test_put_and_get(self, cache):
        cache.put("abc123", "hello world")
        assert cache.get("abc123") == "hello world"

    def test_miss(self, cache):
        assert cache.get("nonexistent") is None

    def test_has(self, cache):
        assert not cache.has("abc123")
        cache.put("abc123", "value")
        assert cache.has("abc123")

    def test_overwrite(self, cache):
        cache.put("abc123", "first")
        cache.put("abc123", "second")
        assert cache.get("abc123") == "second"

    def test_stats_empty(self, cache):
        s = cache.stats()
        assert s["entries"] == 0
        assert s["size_bytes"] == 0

    def test_stats_with_entries(self, cache):
        cache.put("aabbcc", "hello")
        cache.put("ddeeff", "world")
        s = cache.stats()
        assert s["entries"] == 2
        assert s["size_bytes"] > 0

    def test_clear(self, cache):
        cache.put("aabbcc", "hello")
        cache.put("ddeeff", "world")
        removed = cache.clear()
        assert removed == 2
        assert cache.get("aabbcc") is None
        assert cache.get("ddeeff") is None
        s = cache.stats()
        assert s["entries"] == 0

    def test_clear_empty(self, cache):
        removed = cache.clear()
        assert removed == 0

    def test_path_structure(self, cache):
        key = "aabbccdd"
        cache.put(key, "test")
        expected_path = cache.cache_dir / "aa" / "bb" / key
        assert expected_path.exists()


class TestMakeCacheKey:
    def test_deterministic(self):
        key1 = make_cache_key(OpType.GREP, {"input": "ctx", "pattern": "test"}, {"ctx": "hash1"})
        key2 = make_cache_key(OpType.GREP, {"input": "ctx", "pattern": "test"}, {"ctx": "hash1"})
        assert key1 == key2

    def test_different_ops(self):
        key1 = make_cache_key(OpType.GREP, {"input": "ctx"}, {"ctx": "hash1"})
        key2 = make_cache_key(OpType.COUNT, {"input": "ctx"}, {"ctx": "hash1"})
        assert key1 != key2

    def test_different_args(self):
        key1 = make_cache_key(OpType.GREP, {"input": "ctx", "pattern": "a"}, {"ctx": "hash1"})
        key2 = make_cache_key(OpType.GREP, {"input": "ctx", "pattern": "b"}, {"ctx": "hash1"})
        assert key1 != key2

    def test_different_inputs(self):
        key1 = make_cache_key(OpType.GREP, {"input": "ctx", "pattern": "a"}, {"ctx": "hash1"})
        key2 = make_cache_key(OpType.GREP, {"input": "ctx", "pattern": "a"}, {"ctx": "hash2"})
        assert key1 != key2

    def test_resolves_bindings(self):
        # When an arg value matches a key in input_hashes, it should resolve
        key1 = make_cache_key(OpType.COUNT, {"input": "ctx"}, {"ctx": "hash_of_ctx"})
        key2 = make_cache_key(OpType.COUNT, {"input": "ctx"}, {"ctx": "different_hash"})
        assert key1 != key2


class TestCacheIntegration:
    def test_evaluator_uses_cache(self, cache):
        evaluator = LightweightEvaluator(cache=cache)
        bindings = {"context": "line one\nline two\nline three"}

        op = Operation(op=OpType.COUNT, args={"input": "context"})

        # First call: miss
        result1 = evaluator.execute(op, bindings)
        assert result1.value == "3"
        assert not result1.cached

        # Second call: hit
        result2 = evaluator.execute(op, bindings)
        assert result2.value == "3"
        assert result2.cached
        assert result2.cache_key == result1.cache_key

    def test_evaluator_without_cache(self):
        evaluator = LightweightEvaluator()
        bindings = {"context": "hello"}

        op = Operation(op=OpType.COUNT, args={"input": "context"})
        result = evaluator.execute(op, bindings)
        assert result.value == "1"
        assert not result.cached

    def test_different_inputs_different_keys(self, cache):
        evaluator = LightweightEvaluator(cache=cache)

        op = Operation(op=OpType.COUNT, args={"input": "context"})

        result1 = evaluator.execute(op, {"context": "a\nb"})
        result2 = evaluator.execute(op, {"context": "a\nb\nc"})

        assert result1.value == "2"
        assert result2.value == "3"
        assert result1.cache_key != result2.cache_key
