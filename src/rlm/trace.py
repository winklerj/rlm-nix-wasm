"""Execution trace recording for RLM orchestrator runs."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class LLMCallTrace(BaseModel):
    """One LLM send/response round-trip."""
    type: Literal["llm_call"] = "llm_call"
    call_number: int
    timestamp: float           # time.time() wall-clock
    elapsed_s: float
    model: str
    input_tokens: int
    output_tokens: int
    user_message: str
    assistant_message: str


class ExploreStepTrace(BaseModel):
    """One explore-mode step."""
    type: Literal["explore_step"] = "explore_step"
    step_number: int
    timestamp: float
    elapsed_s: float
    operation_op: str          # OpType.value
    operation_args: dict       # type: ignore[type-arg]
    operation_bind: str | None
    result_value: str
    cached: bool
    error: str | None = None


class CommitOperationTrace(BaseModel):
    """One operation within a commit plan."""
    index: int
    operation_op: str
    operation_args: dict       # type: ignore[type-arg]
    operation_bind: str | None
    elapsed_s: float
    result_value: str
    error: str | None = None
    child_trace_ids: list[int] = []


class CommitCycleTrace(BaseModel):
    """One full commit cycle."""
    type: Literal["commit_cycle"] = "commit_cycle"
    cycle_number: int
    timestamp: float
    output_variable: str
    operations: list[CommitOperationTrace]
    result_value: str


class FinalAnswerTrace(BaseModel):
    """The final answer event."""
    type: Literal["final_answer"] = "final_answer"
    timestamp: float
    answer: str
    total_explore_steps: int
    total_commit_cycles: int


TraceEvent = Annotated[
    LLMCallTrace | ExploreStepTrace | CommitCycleTrace | FinalAnswerTrace,
    Field(discriminator="type"),
]


class OrchestratorTrace(BaseModel):
    """Trace for one orchestrator run() invocation."""
    trace_id: int
    depth: int
    query: str
    context_length: int
    model: str
    elapsed_s: float = 0.0
    events: list[TraceEvent] = []
    children: list["OrchestratorTrace"] = []


class ExecutionTrace(BaseModel):
    """Top-level trace document."""
    version: str = "1.1"
    timestamp: str
    root: OrchestratorTrace


class TraceCollector:
    """Thread-safe trace event collector. No-op when disabled."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._lock = threading.Lock()
        self._next_id = 0

    def next_trace_id(self) -> int:
        """Allocate a sequential trace ID. Returns -1 when disabled."""
        if not self.enabled:
            return -1
        with self._lock:
            tid = self._next_id
            self._next_id += 1
            return tid

    def record_llm_call(
        self,
        node: OrchestratorTrace,
        *,
        call_number: int,
        elapsed_s: float,
        model: str,
        input_tokens: int,
        output_tokens: int,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """Record an LLM send/response pair."""
        if not self.enabled:
            return
        with self._lock:
            node.events.append(LLMCallTrace(
                call_number=call_number,
                timestamp=time.time(),
                elapsed_s=elapsed_s,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                user_message=user_message,
                assistant_message=assistant_message,
            ))

    def record_explore_step(
        self,
        node: OrchestratorTrace,
        *,
        step_number: int,
        elapsed_s: float,
        op_type: str,
        op_args: dict,  # type: ignore[type-arg]
        op_bind: str | None,
        result_value: str,
        cached: bool,
        error: str | None = None,
    ) -> None:
        """Record an explore-mode step."""
        if not self.enabled:
            return
        with self._lock:
            node.events.append(ExploreStepTrace(
                step_number=step_number,
                timestamp=time.time(),
                elapsed_s=elapsed_s,
                operation_op=op_type,
                operation_args=op_args,
                operation_bind=op_bind,
                result_value=result_value,
                cached=cached,
                error=error,
            ))

    def record_commit_cycle(
        self,
        node: OrchestratorTrace,
        *,
        cycle_number: int,
        output_variable: str,
        operations: list[CommitOperationTrace],
        result_value: str,
    ) -> None:
        """Record a complete commit cycle."""
        if not self.enabled:
            return
        with self._lock:
            node.events.append(CommitCycleTrace(
                cycle_number=cycle_number,
                timestamp=time.time(),
                output_variable=output_variable,
                operations=operations,
                result_value=result_value,
            ))

    def record_final_answer(
        self,
        node: OrchestratorTrace,
        *,
        answer: str,
        explore_steps: int,
        commit_cycles: int,
    ) -> None:
        """Record the final answer event."""
        if not self.enabled:
            return
        with self._lock:
            node.events.append(FinalAnswerTrace(
                timestamp=time.time(),
                answer=answer,
                total_explore_steps=explore_steps,
                total_commit_cycles=commit_cycles,
            ))

    @staticmethod
    def write_trace(trace: ExecutionTrace, path: Path) -> None:
        """Write the trace to a JSON file."""
        path.write_text(trace.model_dump_json(indent=2))
