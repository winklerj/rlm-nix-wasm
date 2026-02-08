"""Recursive operations — require orchestrator integration."""

from __future__ import annotations

import json
from collections import Counter


def op_combine(args: dict, bindings: dict[str, str]) -> str:  # type: ignore[type-arg]
    """Combine multiple results using a strategy."""
    inputs_ref = args["inputs"]
    strategy = args.get("strategy", "concat")

    # inputs can be a JSON array or a binding name
    if isinstance(inputs_ref, list):
        values = [bindings[ref] for ref in inputs_ref]
    else:
        raw = bindings[inputs_ref]
        values = json.loads(raw) if raw.startswith("[") else [raw]

    if strategy == "concat":
        return "\n".join(values)
    elif strategy == "sum":
        total = sum(int(v.strip()) for v in values if v.strip().isdigit())
        return str(total)
    elif strategy == "vote":
        counts = Counter(v.strip() for v in values)
        return counts.most_common(1)[0][0]
    else:
        # Custom strategy = prompt for an LLM call (handled by orchestrator)
        return "\n".join(values)
