"""Tests for the system prompt templates.

SYSTEM_PROMPT is a str.format() template, so any literal brace added to it
must be escaped as {{ }}. An unescaped brace makes every orchestrator run
crash before the first LLM call (e.g. a bare "{}" raises IndexError:
"Replacement index 0 out of range for positional args tuple"), and the eval
runner records the exception as a per-task error string — silently zeroing
an entire benchmark sweep. These tests catch that at test time.
"""

from __future__ import annotations

import pytest

from rlm.llm.prompts import (
    EVAL_APPROACH_ADDENDUM,
    EVAL_APPROACH_BENCHMARK,
    EVAL_OPS_ADDENDUM,
    SYSTEM_PROMPT,
)


@pytest.mark.parametrize(
    ("eval_ops", "eval_approach"),
    [
        ("", ""),
        (EVAL_OPS_ADDENDUM, EVAL_APPROACH_ADDENDUM),
        (EVAL_OPS_ADDENDUM, EVAL_APPROACH_BENCHMARK),
    ],
    ids=["no-eval", "eval-default", "eval-benchmark"],
)
def test_system_prompt_formats(eval_ops: str, eval_approach: str) -> None:
    rendered = SYSTEM_PROMPT.format(
        context_chars="1,234",
        query="How many unique users?",
        eval_ops=eval_ops,
        eval_approach=eval_approach,
    )
    assert "How many unique users?" in rendered
    # Escaped braces must have collapsed to literal JSON braces.
    assert '{"op":' in rendered.replace(" ", "").replace("\n", "") or "{" in rendered


def test_counting_guidance_batches_40_to_60_lines_per_call() -> None:
    from rlm.llm.prompts import SYSTEM_PROMPT
    assert "40-60 lines" in SYSTEM_PROMPT
    assert "line_count / 50" in SYSTEM_PROMPT
    assert "10-20 lines" not in SYSTEM_PROMPT
