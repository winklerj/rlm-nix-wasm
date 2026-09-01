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


def test_counting_guidance_labels_items_and_tallies_in_eval() -> None:
    from rlm.llm.prompts import SYSTEM_PROMPT
    # Sub-LLMs label reliably but count unreliably: count via per-item labels.
    assert "never ask a \\\nsub-LLM for a count" in SYSTEM_PROMPT or "never ask a sub-LLM for a count" in SYSTEM_PROMPT
    assert "<item number>: <label>" in SYSTEM_PROMPT
    assert "Counter" in SYSTEM_PROMPT
    # Comparison questions: one map for both labels, explicit tie rule.
    assert "same frequency as" in SYSTEM_PROMPT
    assert "within 3%" in SYSTEM_PROMPT


def test_counting_example_is_valid_json_and_tallies() -> None:
    import json
    import re
    from collections import Counter

    from rlm.llm.prompts import SYSTEM_PROMPT
    rendered = SYSTEM_PROMPT.format(eval_ops="", eval_approach="", query="q")
    example = rendered[rendered.index("## Example: Counting pattern"):rendered.index("## Rules")]
    plan = json.loads(example[example.index("{"):].strip())
    ops = {op["op"]: op for op in plan["operations"]}
    code = ops["eval"]["args"]["code"]
    labels = ["1: location\n2: other\n3: Location", "1: other\n2: location"]
    ns: dict[str, object] = {"labels": labels, "re": re, "Counter": Counter}
    exec(code, ns)  # noqa: S102 — our own example
    assert ns["result"] == {"location": 3, "all": {"location": 3, "other": 2}}


def test_counting_example_normalises_shortened_labels() -> None:
    import json
    import re
    from collections import Counter

    from rlm.llm.prompts import SYSTEM_PROMPT
    rendered = SYSTEM_PROMPT.format(eval_ops="", eval_approach="", query="q")
    example = rendered[rendered.index("## Example: Counting pattern"):rendered.index("## Rules")]
    plan = json.loads(example[example.index("{"):].strip())
    code = {op["op"]: op for op in plan["operations"]}["eval"]["args"]["code"]
    labels = ["1: loc\n2: **other**\n3: Location.", "1: other\n2: skip"]
    ns: dict[str, object] = {"labels": labels, "re": re, "Counter": Counter}
    exec(code, ns)  # noqa: S102 — our own example
    assert ns["result"] == {"location": 2, "all": {"location": 2, "other": 2, "skip": 1}}


def test_counting_guidance_defines_labels_and_normalises() -> None:
    from rlm.llm.prompts import SYSTEM_PROMPT
    assert "one-line definition" in SYSTEM_PROMPT
    assert "classification criterion" in SYSTEM_PROMPT
    assert "Normalise each output label" in SYSTEM_PROMPT


def test_counting_guidance_recheck_pass_for_close_tallies() -> None:
    # Calibrated on Qwen3.8: one labeling pass tops out at 0.905 accuracy with
    # errors pooling in the vaguest labels; re-checking just those lines lifts
    # it to 0.930. The guidance must teach the re-check for close comparisons.
    from rlm.llm.prompts import SYSTEM_PROMPT
    assert "RE-CHECK PASS" in SYSTEM_PROMPT
    assert "within ~10%" in SYSTEM_PROMPT
    assert "re-check each against ALL" in SYSTEM_PROMPT


def test_counting_example_map_prompt_uses_full_label_set_with_definitions() -> None:
    # A binary "X or other" map prompt collapses on this model (recall ~0.02) and
    # roots copy the example over the prose, so the example must show the full
    # label set, the header's criterion and a definition + example per label.
    import json

    from rlm.llm.prompts import SYSTEM_PROMPT
    rendered = SYSTEM_PROMPT.format(eval_ops="", eval_approach="", query="q")
    example = rendered[rendered.index("## Example: Counting pattern"):rendered.index("## Rules")]
    plan = json.loads(example[example.index("{"):].strip())
    prompt = {op["op"]: op for op in plan["operations"]}["map"]["args"]["prompt"]
    assert "TYPE OF ITS ANSWER" in prompt
    for label in ("location", "human being", "numeric value", "entity",
                  "abbreviation", "description and abstract concept"):
        assert f"'{label}' =" in prompt
    assert prompt.count("e.g.") >= 6
    assert "'location' or 'other'" not in prompt
