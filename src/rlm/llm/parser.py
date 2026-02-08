"""Parse LLM output into typed actions."""

from __future__ import annotations

import json
import re

from rlm.types import (
    CommitPlan,
    ExploreAction,
    FinalAnswer,
    LLMAction,
    Operation,
    OpType,
)


class ParseError(Exception):
    """Failed to parse LLM output."""
    pass


def parse_llm_output(raw: str) -> LLMAction:
    """Parse raw LLM output into a typed action.

    Handles JSON possibly wrapped in markdown code fences.
    """
    # Strip markdown code fences if present
    cleaned = raw.strip()
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ParseError(f"Invalid JSON from LLM: {e}\nRaw output:\n{raw}")

    mode = data.get("mode")
    if mode == "explore":
        op_data = data["operation"]
        return ExploreAction(
            operation=Operation(
                op=OpType(op_data["op"]),
                args=op_data.get("args", {}),
                bind=op_data.get("bind"),
            )
        )
    elif mode == "commit":
        operations = [
            Operation(
                op=OpType(op_data["op"]),
                args=op_data.get("args", {}),
                bind=op_data.get("bind"),
            )
            for op_data in data["operations"]
        ]
        return CommitPlan(operations=operations, output=data["output"])
    elif mode == "final":
        return FinalAnswer(answer=data["answer"])
    else:
        raise ParseError(f"Unknown mode: {mode}")
