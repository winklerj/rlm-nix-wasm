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


def _clean_name(name: object) -> str | None:
    """Normalize a variable name emitted by the LLM.

    Models sometimes wrap bind/output names in literal quotes or whitespace
    (e.g. ``output: ' "labels"'``), which then fails the bindings lookup and
    burns a whole cycle on an avoidable error.
    """
    if name is None:
        return None
    cleaned = str(name).strip().strip("'\"").strip()
    return cleaned or None


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
    except json.JSONDecodeError:
        # Fallback: try to extract first JSON object from raw output
        m = re.search(r'\{.*?\}', raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError as e2:
                raise ParseError(f"Invalid JSON from LLM: {e2}\nRaw output:\n{raw}")
        else:
            raise ParseError(f"Invalid JSON from LLM: JSONDecodeError\nRaw output:\n{raw}")

    if not isinstance(data, dict):
        raise ParseError(f"Expected JSON object, got {type(data).__name__}")

    mode = data.get("mode")
    if mode == "explore":
        op_data = data.get("operation")
        if op_data is None:
            # Recover from flat structure: {"mode": "explore", "op": "slice", "args": {...}}
            if "op" in data:
                op_data = {"op": data["op"], "args": data.get("args", {}), "bind": data.get("bind")}
            else:
                raise ParseError(
                    'Explore mode requires an "operation" key with "op", "args", and optional "bind". '
                    'Example: {"mode": "explore", "operation": {"op": "slice", "args": {"input": "context", "start": 0, "end": 100}, "bind": "peek"}}'
                )
        if "op" not in op_data:
            raise ParseError(
                'Operation object is missing the "op" field. '
                'Expected: {"op": "<operation_name>", "args": {...}, "bind": "<variable_name>"}'
            )
        return ExploreAction(
            operation=Operation(
                op=OpType(op_data["op"]),
                args=op_data.get("args", {}),
                bind=_clean_name(op_data.get("bind")),
            )
        )
    elif mode == "commit":
        ops_data = data.get("operations")
        if ops_data is None:
            raise ParseError(
                'Commit mode requires an "operations" array and an "output" variable name. '
                'Example: {"mode": "commit", "operations": [{"op": "chunk", "args": {...}, "bind": "chunks"}], "output": "chunks"}'
            )
        operations = [
            Operation(
                op=OpType(op_data["op"]),
                args=op_data.get("args", {}),
                bind=_clean_name(op_data.get("bind")),
            )
            for op_data in ops_data
        ]
        # Default output to last operation's bind if not specified
        output = _clean_name(data.get("output"))
        if not output and operations:
            output = operations[-1].bind or "result"
        return CommitPlan(operations=operations, output=output or "result")
    elif mode == "final":
        answer = data.get("answer")
        if answer is None:
            raise ParseError(
                'Final mode requires an "answer" field. '
                'Example: {"mode": "final", "answer": "your answer here"}'
            )
        return FinalAnswer(answer=answer)
    else:
        raise ParseError(f"Unknown mode: {mode}")
