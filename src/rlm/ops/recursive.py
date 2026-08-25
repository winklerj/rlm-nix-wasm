"""Recursive operations — require orchestrator integration."""

from __future__ import annotations

import re
from collections import Counter

from rlm.ops.values import parse_list_value


def op_combine(args: dict, bindings: dict[str, str]) -> str:  # type: ignore[type-arg]
    """Combine multiple results using a strategy."""
    inputs_ref = args["inputs"]
    strategy = args.get("strategy", "concat")

    # inputs can be a JSON array or a binding name
    if isinstance(inputs_ref, list):
        values = [bindings[ref] for ref in inputs_ref]
    else:
        raw = bindings[inputs_ref]
        values = parse_list_value(raw)

    if strategy == "concat":
        return "\n".join(values)
    elif strategy == "sum":
        # Sub-LLM replies are rarely bare digits ("Count: 3", "**3**", "3\n").
        # Take the first integer in each value; skip values with none rather
        # than silently summing to 0, which makes the caller retry the plan.
        total = 0
        for v in values:
            m = re.search(r"-?\d+", v)
            if m:
                total += int(m.group())
        return str(total)
    elif strategy == "vote":
        counts = Counter(v.strip() for v in values)
        return counts.most_common(1)[0][0]
    else:
        # Custom strategy = prompt for an LLM call (handled by orchestrator)
        return "\n".join(values)
