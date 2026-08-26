"""Recursive operations — require orchestrator integration."""

from __future__ import annotations

import re
from collections import Counter

from rlm.ops.values import parse_list_value


_LABELED_INT = re.compile(
    r"(?:answer|count|total|result)\s*[:=]?\s*\**\s*(-?\d+)", re.IGNORECASE
)


def extract_int(value: str) -> int | None:
    """Pull the integer a sub-LLM reply is reporting, or None if there is none.

    Replies are rarely bare digits ("Count: 3", "**3**", "There are 45 lines,
    12 match"). Prefer a bare integer, then a labeled one ("Answer: 12"), then
    the last integer in the text — models state their conclusion last.
    """
    stripped = value.strip().strip("*`")
    if re.fullmatch(r"-?\d+", stripped):
        return int(stripped)
    labeled = _LABELED_INT.search(value)
    if labeled:
        return int(labeled.group(1))
    found = re.findall(r"-?\d+", value)
    return int(found[-1]) if found else None


def _resolve_inputs(inputs_ref: object, bindings: dict[str, str]) -> list[str]:
    """Resolve combine's ``inputs`` to a flat list of values.

    ``inputs`` may be a binding name or a list of binding names. A referenced
    binding that itself holds a list (the output of ``map``/``chunk``/``split``)
    is flattened, so ``{"inputs": ["counts"]}`` sums every map result rather
    than treating the whole list as one value.
    """
    refs = inputs_ref if isinstance(inputs_ref, list) else [inputs_ref]
    values: list[str] = []
    for ref in refs:
        values.extend(parse_list_value(bindings[ref]))
    return values


def op_combine(args: dict, bindings: dict[str, str]) -> str:  # type: ignore[type-arg]
    """Combine multiple results using a strategy."""
    strategy = args.get("strategy", "concat")
    values = _resolve_inputs(args["inputs"], bindings)

    if strategy == "concat":
        return "\n".join(values)
    elif strategy == "sum":
        # Skip values with no integer rather than silently summing to 0,
        # which makes the caller retry the plan.
        return str(sum(n for n in map(extract_int, values) if n is not None))
    elif strategy == "vote":
        counts = Counter(v.strip() for v in values)
        return counts.most_common(1)[0][0]
    else:
        # Custom strategy = prompt for an LLM call (handled by orchestrator)
        return "\n".join(values)
