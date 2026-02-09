"""Tests for execution trace recording."""

import json
import threading
from pathlib import Path

from rlm.trace import (
    CommitCycleTrace,
    CommitOperationTrace,
    ExecutionTrace,
    ExploreStepTrace,
    FinalAnswerTrace,
    LLMCallTrace,
    OrchestratorTrace,
    TraceCollector,
)


def _make_node(trace_id: int = 0) -> OrchestratorTrace:
    return OrchestratorTrace(
        trace_id=trace_id, depth=0, query="test", context_length=100, model="m",
    )


# --- TraceCollector disabled ---

def test_disabled_collector_noop():
    tc = TraceCollector(enabled=False)
    assert tc.next_trace_id() == -1
    node = _make_node()
    tc.record_llm_call(
        node, call_number=1, elapsed_s=1.0, model="m",
        input_tokens=10, output_tokens=5,
        user_message="hi", assistant_message="hello",
    )
    tc.record_explore_step(
        node, step_number=1, elapsed_s=0.1, op_type="grep",
        op_args={"pattern": "x"}, op_bind="r", result_value="found",
        cached=False,
    )
    tc.record_commit_cycle(
        node, cycle_number=1, output_variable="out",
        operations=[], result_value="done",
    )
    tc.record_final_answer(
        node, answer="42", explore_steps=1, commit_cycles=0,
    )
    assert node.llm_calls == []
    assert node.explore_steps == []
    assert node.commit_cycles == []
    assert node.final_answer is None


# --- TraceCollector enabled ---

def test_sequential_trace_ids():
    tc = TraceCollector(enabled=True)
    assert tc.next_trace_id() == 0
    assert tc.next_trace_id() == 1
    assert tc.next_trace_id() == 2


def test_record_llm_call():
    tc = TraceCollector(enabled=True)
    node = _make_node()
    tc.record_llm_call(
        node, call_number=1, elapsed_s=2.5, model="gpt-5-nano",
        input_tokens=100, output_tokens=50,
        user_message="Begin.", assistant_message='{"mode":"final","answer":"ok"}',
    )
    assert len(node.llm_calls) == 1
    call = node.llm_calls[0]
    assert call.call_number == 1
    assert call.elapsed_s == 2.5
    assert call.model == "gpt-5-nano"
    assert call.input_tokens == 100
    assert call.output_tokens == 50
    assert call.user_message == "Begin."
    assert call.assistant_message == '{"mode":"final","answer":"ok"}'
    assert call.timestamp > 0


def test_record_explore_step():
    tc = TraceCollector(enabled=True)
    node = _make_node()
    tc.record_explore_step(
        node, step_number=3, elapsed_s=0.05, op_type="grep",
        op_args={"input": "context", "pattern": "error"},
        op_bind="matches", result_value="line1\nline2", cached=True,
    )
    assert len(node.explore_steps) == 1
    step = node.explore_steps[0]
    assert step.step_number == 3
    assert step.operation_op == "grep"
    assert step.operation_args == {"input": "context", "pattern": "error"}
    assert step.operation_bind == "matches"
    assert step.result_value == "line1\nline2"
    assert step.cached is True
    assert step.error is None


def test_record_explore_step_with_error():
    tc = TraceCollector(enabled=True)
    node = _make_node()
    tc.record_explore_step(
        node, step_number=1, elapsed_s=0.01, op_type="slice",
        op_args={"input": "x"}, op_bind=None,
        result_value="", cached=False, error="KeyError: x",
    )
    assert node.explore_steps[0].error == "KeyError: x"


def test_record_commit_cycle():
    tc = TraceCollector(enabled=True)
    node = _make_node()
    ops = [
        CommitOperationTrace(
            index=1, operation_op="chunk", operation_args={"input": "ctx", "n": 3},
            operation_bind="chunks", elapsed_s=0.01, result_value="[...]",
        ),
        CommitOperationTrace(
            index=2, operation_op="rlm_call",
            operation_args={"query": "q", "context": "chunks"},
            operation_bind="result", elapsed_s=3.0, result_value="answer",
            child_trace_ids=[1],
        ),
    ]
    tc.record_commit_cycle(
        node, cycle_number=1, output_variable="result",
        operations=ops, result_value="answer",
    )
    assert len(node.commit_cycles) == 1
    cycle = node.commit_cycles[0]
    assert cycle.cycle_number == 1
    assert cycle.output_variable == "result"
    assert len(cycle.operations) == 2
    assert cycle.operations[1].child_trace_ids == [1]
    assert cycle.result_value == "answer"


def test_record_final_answer():
    tc = TraceCollector(enabled=True)
    node = _make_node()
    tc.record_final_answer(
        node, answer="The answer is 42", explore_steps=5, commit_cycles=2,
    )
    assert node.final_answer is not None
    assert node.final_answer.answer == "The answer is 42"
    assert node.final_answer.total_explore_steps == 5
    assert node.final_answer.total_commit_cycles == 2
    assert node.final_answer.timestamp > 0


# --- Serialization ---

def test_execution_trace_json_roundtrip():
    node = _make_node()
    node.final_answer = FinalAnswerTrace(
        timestamp=1000.0, answer="ok", total_explore_steps=1, total_commit_cycles=0,
    )
    node.llm_calls.append(LLMCallTrace(
        call_number=1, timestamp=999.0, elapsed_s=1.0, model="m",
        input_tokens=10, output_tokens=5, user_message="hi", assistant_message="bye",
    ))
    trace = ExecutionTrace(timestamp="2026-02-09T00:00:00Z", root=node)
    raw = trace.model_dump_json(indent=2)
    parsed = json.loads(raw)
    assert parsed["version"] == "1.0"
    assert parsed["root"]["trace_id"] == 0
    assert len(parsed["root"]["llm_calls"]) == 1
    # Round-trip back to model
    restored = ExecutionTrace.model_validate_json(raw)
    assert restored.root.final_answer is not None
    assert restored.root.final_answer.answer == "ok"


def test_nested_children_serialize():
    parent = _make_node(trace_id=0)
    child = _make_node(trace_id=1)
    child.depth = 1
    child.query = "sub-query"
    grandchild = _make_node(trace_id=2)
    grandchild.depth = 2
    child.children.append(grandchild)
    parent.children.append(child)
    trace = ExecutionTrace(timestamp="2026-02-09T00:00:00Z", root=parent)
    raw = trace.model_dump_json()
    restored = ExecutionTrace.model_validate_json(raw)
    assert len(restored.root.children) == 1
    assert len(restored.root.children[0].children) == 1
    assert restored.root.children[0].children[0].trace_id == 2


def test_write_trace(tmp_path: Path):
    node = _make_node()
    trace = ExecutionTrace(timestamp="2026-02-09T00:00:00Z", root=node)
    out = tmp_path / "trace.json"
    TraceCollector.write_trace(trace, out)
    assert out.exists()
    restored = ExecutionTrace.model_validate_json(out.read_text())
    assert restored.root.query == "test"


# --- Thread safety ---

def test_concurrent_record_llm_calls():
    tc = TraceCollector(enabled=True)
    node = _make_node()
    n_threads = 8
    calls_per_thread = 50

    def worker(thread_id: int) -> None:
        for i in range(calls_per_thread):
            tc.record_llm_call(
                node, call_number=thread_id * 1000 + i,
                elapsed_s=0.001, model="m",
                input_tokens=1, output_tokens=1,
                user_message=f"t{thread_id}-{i}", assistant_message="ok",
            )

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(node.llm_calls) == n_threads * calls_per_thread
