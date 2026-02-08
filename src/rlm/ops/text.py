"""Text manipulation operations."""

from __future__ import annotations

import json
import math
import re


def op_slice(args: dict, bindings: dict[str, str]) -> str:  # type: ignore[type-arg]
    """Return a substring of the input."""
    input_text = bindings[args["input"]]
    start = args.get("start", 0)
    end = args.get("end", len(input_text))
    return input_text[start:end]


def op_grep(args: dict, bindings: dict[str, str]) -> str:  # type: ignore[type-arg]
    """Return all lines matching a pattern."""
    input_text = bindings[args["input"]]
    pattern = args["pattern"]
    lines = input_text.split("\n")
    matched = [line for line in lines if re.search(pattern, line)]
    return "\n".join(matched)


def op_count(args: dict, bindings: dict[str, str]) -> str:  # type: ignore[type-arg]
    """Count lines in the input."""
    input_text = bindings[args["input"]]
    mode = args.get("mode", "lines")
    if mode == "chars":
        return str(len(input_text))
    # default: lines
    return str(len(input_text.strip().split("\n"))) if input_text.strip() else "0"


def op_split(args: dict, bindings: dict[str, str]) -> str:  # type: ignore[type-arg]
    """Split input on a delimiter. Returns JSON array."""
    input_text = bindings[args["input"]]
    delimiter = args.get("delimiter", "\n")
    parts = input_text.split(delimiter)
    return json.dumps(parts)


def op_chunk(args: dict, bindings: dict[str, str]) -> str:  # type: ignore[type-arg]
    """Split input into n roughly equal pieces. Returns JSON array."""
    input_text = bindings[args["input"]]
    n = args["n"]
    lines = input_text.split("\n")
    chunk_size = math.ceil(len(lines) / n)
    chunks = []
    for i in range(0, len(lines), chunk_size):
        chunks.append("\n".join(lines[i:i + chunk_size]))
    return json.dumps(chunks)
