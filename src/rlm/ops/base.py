"""Base types for DSL operations."""

from __future__ import annotations

from typing import Protocol


class OpExecutor(Protocol):
    """Protocol for operation executors."""
    def execute(self, args: dict, bindings: dict[str, str]) -> str:  # type: ignore[type-arg]
        """Execute an operation with the given arguments and variable bindings."""
        ...
