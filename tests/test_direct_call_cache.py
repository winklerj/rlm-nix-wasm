"""Leaf (direct) LLM calls are cached content-addressed across repeats."""

from __future__ import annotations

from unittest.mock import patch

from rlm.orchestrator import RLMOrchestrator
from rlm.types import RLMConfig


def _orch(tmp_path) -> RLMOrchestrator:  # type: ignore[no-untyped-def]
    config = RLMConfig(model="test-model", use_nix=False, cache_dir=tmp_path / "cache")
    return RLMOrchestrator(config)


def test_repeated_direct_call_hits_cache(tmp_path) -> None:  # type: ignore[no-untyped-def]
    orch = _orch(tmp_path)
    with patch("rlm.orchestrator.LLMClient.send", return_value="answer A") as send:
        first = orch._direct_call("q", "ctx")
        second = orch._direct_call("q", "ctx")
    assert first == second == "answer A"
    assert send.call_count == 1  # second call served from cache


def test_different_query_or_context_misses_cache(tmp_path) -> None:  # type: ignore[no-untyped-def]
    orch = _orch(tmp_path)
    with patch("rlm.orchestrator.LLMClient.send", side_effect=["A", "B", "C"]) as send:
        assert orch._direct_call("q", "ctx") == "A"
        assert orch._direct_call("q2", "ctx") == "B"
        assert orch._direct_call("q", "ctx2") == "C"
    assert send.call_count == 3


def test_cache_shared_across_orchestrators_with_same_cache_dir(tmp_path) -> None:  # type: ignore[no-untyped-def]
    a = _orch(tmp_path)
    b = _orch(tmp_path)
    with patch("rlm.orchestrator.LLMClient.send", return_value="X") as send:
        a._direct_call("q", "ctx")
        b._direct_call("q", "ctx")
    assert send.call_count == 1
