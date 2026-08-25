"""LLMClient forwards generation limits to the completion call."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from rlm.llm.client import LLMClient
from rlm.types import RLMConfig


def _plain_response(content: str) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(finish_reason="stop", message=message)
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1)
    return SimpleNamespace(choices=[choice], usage=usage)


def _send_and_capture(config: RLMConfig, use_tools: bool = True) -> dict:  # type: ignore[type-arg]
    client = LLMClient(config, use_tools=use_tools)
    client.set_system_prompt("sys")
    with patch("rlm.llm.client.completion", return_value=_plain_response("ok")) as mocked:
        client.send("hi")
    return mocked.call_args.kwargs


def test_max_output_tokens_forwarded_as_max_tokens() -> None:
    kw = _send_and_capture(RLMConfig(model="m", use_nix=False, max_output_tokens=16384))
    assert kw["max_tokens"] == 16384


def test_max_output_tokens_applies_to_direct_calls_too() -> None:
    kw = _send_and_capture(
        RLMConfig(model="m", use_nix=False, max_output_tokens=100), use_tools=False
    )
    assert kw["max_tokens"] == 100
    assert "tools" not in kw


def test_no_cap_by_default() -> None:
    kw = _send_and_capture(RLMConfig(model="m", use_nix=False))
    assert "max_tokens" not in kw


def test_tool_call_missing_bind_does_not_crash() -> None:
    """Models sometimes omit schema-required fields; the client must not KeyError."""
    tc = SimpleNamespace(function=SimpleNamespace(
        name="rlm_explore",
        arguments='{"op": "count", "args": {"input": "context", "mode": "lines"}}',
    ))
    message = SimpleNamespace(content=None, tool_calls=[tc])
    choice = SimpleNamespace(finish_reason="tool_calls", message=message)
    resp = SimpleNamespace(choices=[choice], usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))
    client = LLMClient(RLMConfig(model="m", use_nix=False))
    client.set_system_prompt("sys")
    with patch("rlm.llm.client.completion", return_value=resp):
        out = client.send("hi")
    import json
    assert json.loads(out)["operation"]["bind"] is None


def test_empty_content_at_length_limit_has_clear_message() -> None:
    from rlm.llm.client import LLMRefusalError
    message = SimpleNamespace(content="", tool_calls=None)
    choice = SimpleNamespace(finish_reason="length", message=message)
    resp = SimpleNamespace(choices=[choice], usage=SimpleNamespace(prompt_tokens=1, completion_tokens=16384))
    client = LLMClient(RLMConfig(model="m", use_nix=False, max_output_tokens=16384))
    client.set_system_prompt("sys")
    with patch("rlm.llm.client.completion", return_value=resp):
        try:
            client.send("hi")
        except LLMRefusalError as e:
            assert "output token limit" in str(e)
        else:
            raise AssertionError("expected LLMRefusalError")


def test_reasoning_strength_forwarded_via_extra_body() -> None:
    kw = _send_and_capture(RLMConfig(model="m", use_nix=False, reasoning_strength="low"))
    assert kw["extra_body"] == {"chat_template_kwargs": {"reasoning_strength": "low"}}
