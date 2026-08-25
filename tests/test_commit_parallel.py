"""Tests for parallel execution of independent rlm_call ops in commit plans."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rlm.orchestrator import RLMOrchestrator
from rlm.types import CommitPlan, Operation, OpType, RLMConfig


@pytest.fixture
def orch(tmp_path):
    config = RLMConfig(model="test-model", use_nix=False, cache_dir=tmp_path / "cache")
    return RLMOrchestrator(config)


def _rlm_call(query: str, context: str, bind: str) -> Operation:
    return Operation(op=OpType.RLM_CALL, args={"query": query, "context": context}, bind=bind)


def test_independent_rlm_calls_are_grouped_and_dependent_ones_are_not(orch):
    plan = CommitPlan(
        operations=[
            _rlm_call("q1", "context", "a"),
            _rlm_call("q2", "context", "b"),
            _rlm_call("q3", "a", "c"),  # depends on `a` produced above
        ],
        output="c",
    )
    group_sizes: list[int] = []
    real = orch._parallel_rlm_calls

    def recording(group, bindings, depth):
        group_sizes.append(len(group))
        return real(group, bindings, depth)

    def fake_spawn(query, context_text, depth):
        return f"{query}:{context_text}", MagicMock()

    with patch.object(orch, "_parallel_rlm_calls", side_effect=recording), \
         patch.object(orch, "_spawn_child", side_effect=fake_spawn):
        result, _traces = orch._execute_commit_plan(plan, {"context": "CTX"}, depth=0)

    assert group_sizes == [2, 1]
    # Order and bindings are preserved; q3 sees the value bound by q1.
    assert result == "q3:q1:CTX"


def test_failed_plan_preserves_bindings_of_successful_ops(orch):
    """A failing final step must not discard earlier (expensive) results."""
    plan = CommitPlan(
        operations=[
            _rlm_call("q1", "context", "expensive"),
            # eval without a sandbox configured raises -> plan fails here
            Operation(op=OpType.EVAL, args={"code": "result = 1"}, bind="agg"),
        ],
        output="agg",
    )
    bindings = {"context": "CTX"}
    with patch.object(orch, "_spawn_child", return_value=("leaf answer", MagicMock())):
        with pytest.raises(RuntimeError) as excinfo:
            orch._execute_commit_plan(plan, bindings, depth=0)
    assert bindings["expensive"] == "leaf answer"
    assert "expensive" in str(excinfo.value)
    assert "preserved" in str(excinfo.value)


def test_single_rlm_call_still_works(orch):
    plan = CommitPlan(operations=[_rlm_call("q", "context", "out")], output="out")
    with patch.object(orch, "_spawn_child", return_value=("answer", MagicMock())):
        result, _ = orch._execute_commit_plan(plan, {"context": "CTX"}, depth=0)
    assert result == "answer"
