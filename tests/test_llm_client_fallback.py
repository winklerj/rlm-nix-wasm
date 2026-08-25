"""LLMClient falls back to a no-tools call when the server rejects tool syntax."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from rlm.llm.client import LLMClient
from rlm.types import RLMConfig


def _plain_response(content: str) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(finish_reason="stop", message=message)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    return SimpleNamespace(choices=[choice], usage=usage)


def test_retries_without_tools_on_peg_format_error() -> None:
    client = LLMClient(RLMConfig(model="test-model", use_nix=False))
    client.set_system_prompt("sys")
    calls: list[dict] = []  # type: ignore[type-arg]

    def fake_completion(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError(
                "InternalServerError: The model produced output that does not "
                "match the expected peg-native format"
            )
        return _plain_response('{"mode": "final", "answer": "42"}')

    with patch("rlm.llm.client.completion", side_effect=fake_completion):
        out = client.send("hello")

    assert out == '{"mode": "final", "answer": "42"}'
    assert len(calls) == 2
    assert "tools" in calls[0] and "tool_choice" in calls[0]
    assert "tools" not in calls[1] and "tool_choice" not in calls[1]


def test_second_failure_adds_format_nudge_and_retries_once_more() -> None:
    client = LLMClient(RLMConfig(model="test-model", use_nix=False))
    client.set_system_prompt("sys")
    calls: list[dict] = []  # type: ignore[type-arg]
    err = RuntimeError("does not match the expected peg-native format")

    def fake_completion(**kwargs):  # type: ignore[no-untyped-def]
        calls.append({"kw": kwargs, "n_msgs": len(kwargs["messages"])})
        if len(calls) <= 2:
            raise err
        return _plain_response('{"mode": "final", "answer": "ok"}')

    with patch("rlm.llm.client.completion", side_effect=fake_completion):
        out = client.send("hello")

    assert out == '{"mode": "final", "answer": "ok"}'
    assert len(calls) == 3
    assert "tools" not in calls[2]["kw"]
    # Third attempt carries one extra (corrective) user message.
    assert calls[2]["n_msgs"] == calls[1]["n_msgs"] + 1
    assert "Do NOT use tool-call syntax" in client.messages[-2]["content"]


def test_three_failures_raise() -> None:
    client = LLMClient(RLMConfig(model="test-model", use_nix=False))
    client.set_system_prompt("sys")
    err = RuntimeError("does not match the expected peg-native format")
    with patch("rlm.llm.client.completion", side_effect=err) as mocked:
        with pytest.raises(RuntimeError, match="peg-native"):
            client.send("hello")
    assert mocked.call_count == 3


def test_unrelated_errors_still_raise() -> None:
    client = LLMClient(RLMConfig(model="test-model", use_nix=False))
    client.set_system_prompt("sys")
    with patch("rlm.llm.client.completion", side_effect=RuntimeError("connection refused")):
        with pytest.raises(RuntimeError, match="connection refused"):
            client.send("hello")


def test_no_tools_client_does_not_retry() -> None:
    client = LLMClient(RLMConfig(model="test-model", use_nix=False), use_tools=False)
    client.set_system_prompt("sys")
    err = RuntimeError("does not match the expected peg-native format")
    with patch("rlm.llm.client.completion", side_effect=err) as mocked:
        with pytest.raises(RuntimeError):
            client.send("hello")
    assert mocked.call_count == 1
