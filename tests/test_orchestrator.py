"""Tests for the RLM orchestrator with mocked LLM client."""

import json
from unittest.mock import MagicMock, patch

import pytest
from pytest import approx

from rlm.orchestrator import RLMOrchestrator
from rlm.types import CommitPlan, Operation, OpType, RLMConfig


@pytest.fixture
def config():
    return RLMConfig(
        model="test-model",
        max_explore_steps=5,
        max_commit_cycles=3,
        max_recursion_depth=1,
        use_nix=False,
        verbose=False,
    )


def _make_orchestrator_with_responses(config, responses):
    """Create an orchestrator with a mocked LLM that returns canned responses."""
    orchestrator = RLMOrchestrator(config)
    response_iter = iter(responses)
    orchestrator.llm.send = MagicMock(side_effect=lambda _: next(response_iter))
    orchestrator.llm.set_system_prompt = MagicMock()
    return orchestrator


class TestFinalAnswer:
    def test_immediate_final(self, config):
        responses = [
            json.dumps({"mode": "final", "answer": "42"}),
        ]
        orch = _make_orchestrator_with_responses(config, responses)
        result = orch.run("What is the answer?", "some context")
        assert result == "42"

    def test_final_after_explore(self, config):
        responses = [
            json.dumps({
                "mode": "explore",
                "operation": {
                    "op": "slice",
                    "args": {"input": "context", "start": 0, "end": 10},
                    "bind": "peek",
                },
            }),
            json.dumps({"mode": "final", "answer": "The context starts with 'some conte'"}),
        ]
        orch = _make_orchestrator_with_responses(config, responses)
        result = orch.run("What does the context start with?", "some context text")
        assert "some conte" in result


class TestExploreMode:
    def test_explore_binds_variable(self, config):
        responses = [
            json.dumps({
                "mode": "explore",
                "operation": {
                    "op": "grep",
                    "args": {"input": "context", "pattern": "hello"},
                    "bind": "matches",
                },
            }),
            json.dumps({
                "mode": "explore",
                "operation": {
                    "op": "count",
                    "args": {"input": "matches"},
                    "bind": "n",
                },
            }),
            json.dumps({"mode": "final", "answer": "1"}),
        ]
        orch = _make_orchestrator_with_responses(config, responses)
        result = orch.run("How many lines contain hello?", "hello world\ngoodbye world")
        assert result == "1"

    def test_max_explore_steps(self, config):
        # Generate more explore steps than allowed (config.max_explore_steps = 5)
        explore = json.dumps({
            "mode": "explore",
            "operation": {
                "op": "count",
                "args": {"input": "context"},
            },
        })
        responses = [explore] * 6 + [
            json.dumps({"mode": "final", "answer": "forced"}),
        ]
        orch = _make_orchestrator_with_responses(config, responses)
        result = orch.run("test", "line1\nline2")
        assert result == "forced"
        # Should have been told about the limit
        calls = [str(c) for c in orch.llm.send.call_args_list]
        assert any("maximum" in c for c in calls)


class TestCommitMode:
    def test_commit_plan_execution(self, config):
        responses = [
            json.dumps({
                "mode": "commit",
                "operations": [
                    {"op": "grep", "args": {"input": "context", "pattern": "a"}, "bind": "filtered"},
                    {"op": "count", "args": {"input": "filtered"}, "bind": "total"},
                ],
                "output": "total",
            }),
            json.dumps({"mode": "final", "answer": "2"}),
        ]
        orch = _make_orchestrator_with_responses(config, responses)
        result = orch.run("How many lines contain 'a'?", "apple\nbanana\ncherry")
        assert result == "2"

    def test_max_commit_cycles(self, config):
        commit = json.dumps({
            "mode": "commit",
            "operations": [
                {"op": "count", "args": {"input": "context"}, "bind": "n"},
            ],
            "output": "n",
        })
        responses = [commit] * 4 + [
            json.dumps({"mode": "final", "answer": "forced"}),
        ]
        orch = _make_orchestrator_with_responses(config, responses)
        result = orch.run("test", "data")
        assert result == "forced"
        calls = [str(c) for c in orch.llm.send.call_args_list]
        assert any("maximum" in c for c in calls)


class TestRecursion:
    def test_depth_limiting(self, config):
        """At max depth, should use direct call instead of explore/commit."""
        config.max_recursion_depth = 0
        orch = RLMOrchestrator(config)

        # Mock the direct LLM call
        with patch.object(orch, '_direct_call', return_value="direct answer") as mock_direct:
            result = orch.run("test", "context", depth=1)
            assert result == "direct answer"
            mock_direct.assert_called_once()

    def test_small_child_context_uses_direct_call(self, config):
        """A child whose context is below min_recursive_chars is a direct call
        even when depth would otherwise allow a full explore/commit loop."""
        config.max_recursion_depth = 5
        config.min_recursive_chars = 100
        orch = RLMOrchestrator(config)

        with patch.object(orch, '_direct_call', return_value="direct answer") as mock_direct:
            result = orch.run("test", "tiny context", depth=1)
            assert result == "direct answer"
            mock_direct.assert_called_once()

    def test_root_never_shortcut_by_size(self, config):
        """The size rule only applies to children: the root (depth 0) always
        runs the explore/commit loop, however small its context."""
        config.min_recursive_chars = 10_000
        orch = _make_orchestrator_with_responses(
            config, [json.dumps({"mode": "final", "answer": "looped"})]
        )
        with patch.object(orch, '_direct_call') as mock_direct:
            assert orch.run("test", "tiny context", depth=0) == "looped"
            mock_direct.assert_not_called()


class TestParseErrorRecovery:
    def test_recovers_from_bad_json(self, config):
        responses = [
            "This is not JSON at all",
            json.dumps({"mode": "final", "answer": "recovered"}),
        ]
        orch = _make_orchestrator_with_responses(config, responses)
        result = orch.run("test", "context")
        assert result == "recovered"

    def test_recovers_from_invalid_mode(self, config):
        responses = [
            json.dumps({"mode": "unknown_mode"}),
            json.dumps({"mode": "final", "answer": "recovered"}),
        ]
        orch = _make_orchestrator_with_responses(config, responses)
        result = orch.run("test", "context")
        assert result == "recovered"


class TestExploreCommitFlow:
    def test_explore_then_commit_then_final(self, config):
        """Full flow: explore, commit, final."""
        responses = [
            # Explore: peek at context
            json.dumps({
                "mode": "explore",
                "operation": {
                    "op": "slice",
                    "args": {"input": "context", "start": 0, "end": 20},
                    "bind": "peek",
                },
            }),
            # Commit: grep and count
            json.dumps({
                "mode": "commit",
                "operations": [
                    {"op": "grep", "args": {"input": "context", "pattern": "line"}, "bind": "lines"},
                    {"op": "count", "args": {"input": "lines"}, "bind": "total"},
                ],
                "output": "total",
            }),
            # Final
            json.dumps({"mode": "final", "answer": "3"}),
        ]
        orch = _make_orchestrator_with_responses(config, responses)
        result = orch.run("How many lines?", "line 1\nline 2\nline 3")
        assert result == "3"


class TestChildModel:
    def test_child_uses_child_model(self, config):
        """Child orchestrator should use child_model when set."""
        config.model = "orchestrator-model"
        config.child_model = "child-model"
        parent = RLMOrchestrator(config)

        with patch.object(RLMOrchestrator, 'run', return_value="result"):
            parent._recursive_call("q", "ctx", depth=0)

        assert len(parent.child_orchestrators) == 1
        child = parent.child_orchestrators[0]
        assert child.config.model == "child-model"
        assert child.config.child_model is None

    def test_child_falls_back_to_parent_model(self, config):
        """Child orchestrator should use parent model when child_model is None."""
        config.model = "orchestrator-model"
        config.child_model = None
        parent = RLMOrchestrator(config)

        with patch.object(RLMOrchestrator, 'run', return_value="result"):
            parent._recursive_call("q", "ctx", depth=0)

        child = parent.child_orchestrators[0]
        assert child.config.model == "orchestrator-model"

    def test_parallel_map_uses_child_model(self, config):
        """Parallel map should create children with child_model."""
        config.model = "orchestrator-model"
        config.child_model = "child-model"
        config.max_parallel_jobs = 2
        parent = RLMOrchestrator(config)

        with patch.object(RLMOrchestrator, 'run', return_value="result"):
            parent._parallel_map("prompt", ["a", "b"], depth=0)

        assert len(parent.child_orchestrators) == 2
        for child in parent.child_orchestrators:
            assert child.config.model == "child-model"

    def test_grandchild_uses_child_model(self, config):
        """Children of children should also use the child model."""
        config.model = "orchestrator-model"
        config.child_model = "child-model"
        parent = RLMOrchestrator(config)

        with patch.object(RLMOrchestrator, 'run', return_value="result"):
            parent._recursive_call("q", "ctx", depth=0)

        child = parent.child_orchestrators[0]
        with patch.object(RLMOrchestrator, 'run', return_value="result"):
            child._recursive_call("q2", "ctx2", depth=1)

        grandchild = child.child_orchestrators[0]
        assert grandchild.config.model == "child-model"

    def test_get_total_cost_dual_model(self, config):
        """get_total_cost should price each orchestrator at its own model rate."""
        config.model = "expensive-model"
        config.child_model = "cheap-model"
        parent = RLMOrchestrator(config)
        parent.llm.total_input_tokens = 1000
        parent.llm.total_output_tokens = 500

        with patch.object(RLMOrchestrator, 'run', return_value="result"):
            parent._recursive_call("q", "ctx", depth=0)
        child = parent.child_orchestrators[0]
        child.llm.total_input_tokens = 2000
        child.llm.total_output_tokens = 1000

        def mock_pricing(model: str, inp: int, out: int) -> float:
            if model == "expensive-model":
                return inp * 0.01 + out * 0.05
            return inp * 0.001 + out * 0.005

        total = parent.get_total_cost(mock_pricing)
        expected_parent = 1000 * 0.01 + 500 * 0.05   # 35.0
        expected_child = 2000 * 0.001 + 1000 * 0.005  # 7.0
        assert total == approx(expected_parent + expected_child)


class TestMapDirect:
    """`map` pieces are single direct calls, never child explore/commit loops."""

    def test_map_pieces_are_direct_calls_even_when_large(self, config):
        config.min_recursive_chars = 10
        config.max_recursion_depth = 1
        orch = RLMOrchestrator(config)
        with patch.object(RLMOrchestrator, '_direct_call', return_value="leaf") as direct:
            out = orch._parallel_map("label these", ["x" * 5000, "y" * 5000], depth=0)
        assert json.loads(out) == ["leaf", "leaf"]
        assert direct.call_count == 2
        # Pieces ran past max depth, i.e. they short-circuited to a direct call.
        assert all(c.trace_node.depth == 2 for c in orch.child_orchestrators)

    def test_map_direct_can_be_disabled(self, config):
        config.min_recursive_chars = 10
        config.max_recursion_depth = 1
        config.map_direct = False
        orch = RLMOrchestrator(config)
        with patch.object(RLMOrchestrator, 'run', return_value="r") as run:
            orch._parallel_map("p", ["x" * 5000], depth=0)
        # Piece runs as a depth-1 child, which is below max depth -> full loop.
        assert run.call_args.kwargs.get("depth", run.call_args.args[-1]) == 1


class TestMapFanoutGuardrail:
    @staticmethod
    def _plan():
        return CommitPlan(operations=[
            Operation(op=OpType.MAP, args={"prompt": "label", "input": "chunks"}, bind="out"),
        ], output="out")

    def test_one_line_per_piece_over_many_pieces_is_refused_before_any_call(self, config):
        orch = RLMOrchestrator(config)
        bindings = {"chunks": json.dumps([f"line {i}" for i in range(65)])}
        with patch.object(RLMOrchestrator, '_direct_call', return_value="leaf") as direct:
            with pytest.raises(RuntimeError, match="map over 65 pieces averaging 1.0 lines"):
                orch._execute_commit_plan(self._plan(), bindings, depth=0)
        assert direct.call_count == 0

    def test_absolute_piece_cap_is_enforced(self, config):
        config.max_map_items = 100
        orch = RLMOrchestrator(config)
        pieces = ["\n".join(f"l{j}" for j in range(50))] * 101
        with patch.object(RLMOrchestrator, '_direct_call', return_value="leaf") as direct:
            with pytest.raises(RuntimeError, match="map over 101 pieces"):
                orch._execute_commit_plan(self._plan(), {"chunks": json.dumps(pieces)}, depth=0)
        assert direct.call_count == 0

    def test_small_map_over_single_lines_is_allowed(self, config):
        orch = RLMOrchestrator(config)
        bindings = {"chunks": json.dumps([f"line {i}" for i in range(15)])}
        with patch.object(RLMOrchestrator, '_direct_call', return_value="leaf"):
            out, _ = orch._execute_commit_plan(self._plan(), bindings, depth=0)
        assert json.loads(out) == ["leaf"] * 15

    def test_sweep_scale_chunking_is_allowed(self, config):
        # 4M tokens ~ 100K lines at n = lines / 50 -> ~2,000 pieces of 50 lines.
        orch = RLMOrchestrator(config)
        pieces = ["\n".join(f"l{j}" for j in range(50))] * 2048
        with patch.object(RLMOrchestrator, '_direct_call', return_value="leaf") as direct:
            orch._execute_commit_plan(self._plan(), {"chunks": json.dumps(pieces)}, depth=0)
        assert direct.call_count == 2048

    def test_rechunked_repeat_of_same_map_is_refused_and_previous_result_kept(self, config):
        orch = RLMOrchestrator(config)
        lines = [f"line {i}" for i in range(100)]
        four = ["\n".join(lines[i:i + 25]) for i in range(0, 100, 25)]
        ten = ["\n".join(lines[i:i + 10]) for i in range(0, 100, 10)]
        bindings: dict[str, str] = {}
        with patch.object(RLMOrchestrator, '_direct_call', return_value="leaf") as direct:
            bindings["chunks"] = json.dumps(four)
            out, _ = orch._execute_commit_plan(self._plan(), bindings, depth=0)
            assert direct.call_count == 4
            bindings = {"chunks": json.dumps(ten)}  # 'out' was an intermediate: gone
            with pytest.raises(RuntimeError, match="already run over this same text as 4 pieces"):
                orch._execute_commit_plan(self._plan(), bindings, depth=0)
        assert direct.call_count == 4
        assert bindings["out"] == out  # preserved for reuse by the model

    def test_identical_repeat_of_same_map_is_allowed(self, config):
        orch = RLMOrchestrator(config)
        pieces = json.dumps(["a\nb", "c\nd"])
        with patch.object(RLMOrchestrator, '_direct_call', return_value="leaf"):
            orch._execute_commit_plan(self._plan(), {"chunks": pieces}, depth=0)
            out, _ = orch._execute_commit_plan(self._plan(), {"chunks": pieces}, depth=0)
        assert json.loads(out) == ["leaf", "leaf"]

    def test_same_prompt_over_different_text_is_allowed(self, config):
        orch = RLMOrchestrator(config)
        with patch.object(RLMOrchestrator, '_direct_call', return_value="leaf") as direct:
            orch._execute_commit_plan(self._plan(), {"chunks": json.dumps(["a\nb"])}, depth=0)
            orch._execute_commit_plan(self._plan(), {"chunks": json.dumps(["a", "b", "c"])}, depth=0)
        assert direct.call_count == 4

    def test_force_allows_rechunking(self, config):
        orch = RLMOrchestrator(config)
        forced = CommitPlan(operations=[
            Operation(op=OpType.MAP, args={"prompt": "label", "input": "chunks", "force": True}, bind="out"),
        ], output="out")
        with patch.object(RLMOrchestrator, '_direct_call', return_value="leaf") as direct:
            orch._execute_commit_plan(self._plan(), {"chunks": json.dumps(["a\nb"])}, depth=0)
            orch._execute_commit_plan(forced, {"chunks": json.dumps(["a", "b"])}, depth=0)
        assert direct.call_count == 3

    def test_verbose_logs_full_map_prompt(self, config):
        config.verbose = True
        orch = RLMOrchestrator(config)
        long_prompt = "Classify by the type of the answer: " + "x" * 200
        plan = CommitPlan(operations=[
            Operation(op=OpType.MAP, args={"prompt": long_prompt, "input": "chunks"}, bind="out"),
        ], output="out")
        with patch.object(RLMOrchestrator, '_direct_call', return_value="leaf"):
            with patch.object(orch.console, 'print') as printed:
                orch._execute_commit_plan(plan, {"chunks": json.dumps(["a", "b"])}, depth=0)
        logged = "\n".join(str(c.args[0]) for c in printed.call_args_list)
        assert long_prompt in logged
