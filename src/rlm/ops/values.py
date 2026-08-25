"""Helpers for interpreting operation output values."""

from __future__ import annotations

import ast
import json


def parse_list_value(raw: str) -> list[str]:
    """Interpret an op result as a list of string items.

    Results flow between ops as strings. List-valued results are usually JSON
    (``chunk``/``split`` emit ``json.dumps``), but ``eval`` prints whatever the
    user code produced — typically a Python ``repr`` such as ``['a', 'b']`` with
    single quotes, which is not JSON. Accept both, and fall back to treating the
    whole value as a single item rather than raising.
    """
    stripped = raw.strip()
    if not stripped.startswith("["):
        return [raw]
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(stripped)  # noqa: S307 — literals only
        except (ValueError, SyntaxError):
            return [raw]
    if isinstance(parsed, (list, tuple)):
        return [item if isinstance(item, str) else str(item) for item in parsed]
    return [raw]
