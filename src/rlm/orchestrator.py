"""RLM Orchestrator — manages the explore/commit protocol loop."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console

from rlm.cache.store import CacheStore
from rlm.evaluator.lightweight import LightweightEvaluator
from rlm.llm.client import LLMClient
from rlm.llm.parser import ParseError, parse_llm_output
from rlm.llm.prompts import (
    EVAL_APPROACH_ADDENDUM,
    EVAL_APPROACH_BENCHMARK,
    EVAL_OPS_ADDENDUM,
    SYSTEM_PROMPT,
)
from rlm.ops.values import parse_list_value
from rlm.timing import TimingProfile
from rlm.trace import CommitOperationTrace, ExecutionTrace, OrchestratorTrace, TraceCollector
from rlm.types import (
    CommitPlan,
    Context,
    ExploreAction,
    FinalAnswer,
    Operation,
    OpType,
    RLMConfig,
)

logger = logging.getLogger(__name__)


class RLMOrchestrator:
    """Orchestrates the explore/commit protocol between the LLM and evaluators."""

    def __init__(self, config: RLMConfig, parent: "RLMOrchestrator | None" = None,
                 trace_collector: TraceCollector | None = None):
        self.config = config
        self.trace_collector = trace_collector or TraceCollector()
        self.trace_node = OrchestratorTrace(
            trace_id=self.trace_collector.next_trace_id(),
            depth=0, query="", context_length=0, model=config.model,
        )
        self.profile = TimingProfile(enabled=config.verbose)
        self.console = Console(stderr=True)
        self.llm = LLMClient(config, profile=self.profile,
                              verbose=config.verbose, console=self.console,
                              trace=self.trace_collector,
                              trace_node=self.trace_node)
        self.cache = CacheStore(config.cache_dir)

        # Initialize Wasm sandbox for eval operations (lazy load)
        wasm_sandbox = None
        if config.wasm_python_path:
            from rlm.evaluator.wasm_sandbox import WasmSandbox
            wasm_sandbox = WasmSandbox(
                python_wasm_path=config.wasm_python_path,
                fuel=config.wasm_fuel,
                memory_mb=config.wasm_memory_mb,
            )

        self.evaluator = LightweightEvaluator(
            cache=self.cache, profile=self.profile, wasm_sandbox=wasm_sandbox,
        )
        self.parent = parent
        self.child_orchestrators: list[RLMOrchestrator] = []

        if config.use_nix:
            from rlm.nix.builder import NixBuilder
            self.nix_builder = NixBuilder(max_jobs=config.max_parallel_jobs)
            if not self.nix_builder.available:
                raise RuntimeError(
                    "Nix is not installed but Nix sandboxing is enabled by default. "
                    "Install Nix from https://nixos.org/ or pass --no-nix to disable."
                )
        else:
            self.nix_builder = None  # type: ignore[assignment]

    def run(self, query: str, context_text: str, depth: int = 0) -> str:
        """Execute an RLM query against a context.

        Args:
            query: The question to answer.
            context_text: The full context text.
            depth: Current recursion depth.

        Returns:
            The final answer as a string.
        """
        self.trace_node.depth = depth
        self.trace_node.query = query
        self.trace_node.context_length = len(context_text)
        run_start = time.monotonic()

        if depth > self.config.max_recursion_depth:
            return self._direct_call(query, context_text)
        # Small child contexts fit in one prompt: an explore/commit loop would
        # only re-discover what a single read gives for free.
        if depth > 0 and len(context_text) < self.config.min_recursive_chars:
            return self._direct_call(query, context_text)

        ctx = Context(content=context_text)
        bindings: dict[str, str] = {"context": ctx.content}

        # Conditionally include eval docs when Wasm sandbox is available
        eval_ops = EVAL_OPS_ADDENDUM if self.config.wasm_python_path else ""
        if self.config.wasm_python_path:
            eval_approach = (
                EVAL_APPROACH_BENCHMARK if self.config.benchmark_eval_prompt
                else EVAL_APPROACH_ADDENDUM
            )
        else:
            eval_approach = ""

        system = SYSTEM_PROMPT.format(
            context_chars=f"{len(ctx.content):,}",
            query=query,
            eval_ops=eval_ops,
            eval_approach=eval_approach,
        )
        self.llm.set_system_prompt(system)

        explore_steps = 0
        commit_cycles = 0
        max_parse_retries = 3
        max_overflow_nudges = 3
        explore_overflow_nudges = 0
        commit_overflow_nudges = 0

        # Give the warm-up away for free: models otherwise spend their first
        # several explore steps rediscovering size, line count and a preview.
        preview_chars = 500
        line_count = ctx.content.count("\n") + 1 if ctx.content else 0
        preview = ctx.content[:preview_chars]
        preview_note = (
            f" (first {preview_chars} chars)" if len(ctx.content) > preview_chars
            else " (entire context)"
        )
        response = self.llm.send(
            "Begin. The context variable is available: "
            f"{len(ctx.content):,} chars, {line_count:,} lines.\n"
            f"Preview{preview_note}:\n{preview}"
        )

        parse_retries = 0
        while True:
            try:
                action = parse_llm_output(response)
                parse_retries = 0  # Reset on success
            except ParseError as e:
                parse_retries += 1
                logger.warning("Parse error (%d/%d): %s", parse_retries, max_parse_retries, e)
                if parse_retries >= max_parse_retries:
                    raise RuntimeError(
                        f"LLM failed to produce valid JSON after {max_parse_retries} attempts. "
                        f"Last error: {e}"
                    )
                response = self.llm.send(
                    f"Your response was not valid JSON. Please respond with a valid JSON "
                    f"object with a 'mode' field. Error: {e}"
                )
                continue

            if isinstance(action, FinalAnswer):
                if self.config.verbose:
                    self.console.print(
                        f"[green]Final answer after {explore_steps} explore steps, "
                        f"{commit_cycles} commit cycles[/green]"
                    )
                self.trace_collector.record_final_answer(
                    self.trace_node, answer=action.answer,
                    explore_steps=explore_steps, commit_cycles=commit_cycles,
                )
                self.trace_node.elapsed_s = time.monotonic() - run_start
                return action.answer

            elif isinstance(action, ExploreAction):
                explore_steps += 1
                if explore_steps > self.config.max_explore_steps:
                    explore_overflow_nudges += 1
                    if explore_overflow_nudges >= max_overflow_nudges:
                        raise RuntimeError(
                            f"LLM did not COMMIT or FINAL after {max_overflow_nudges} nudges "
                            f"past the {self.config.max_explore_steps} explore step limit."
                        )
                    response = self.llm.send(
                        f"You have reached the maximum of {self.config.max_explore_steps} "
                        f"explore steps. Please COMMIT a plan or provide a FINAL answer."
                    )
                    continue

                op = action.operation
                try:
                    step_start = time.monotonic()
                    result = self.evaluator.execute(op, bindings)
                    step_elapsed = time.monotonic() - step_start

                    if op.bind:
                        bindings[op.bind] = result.value

                    if self.config.verbose:
                        op_desc = self._format_op(op)
                        cache_note = ", cached" if result.cached else ""
                        bind_note = f" → {op.bind}" if op.bind else ""
                        self.console.print(
                            f"[dim]EXPLORE step {explore_steps}/{self.config.max_explore_steps}: "
                            f"{op_desc}{bind_note}  ({step_elapsed:.3f}s{cache_note})[/dim]"
                        )

                    self.trace_collector.record_explore_step(
                        self.trace_node, step_number=explore_steps,
                        elapsed_s=step_elapsed, op_type=op.op.value,
                        op_args=op.args, op_bind=op.bind,
                        result_value=result.value, cached=result.cached,
                    )

                    display_value = result.value
                    if len(display_value) > self.config.max_result_chars:
                        display_value = (
                            display_value[:self.config.max_result_chars]
                            + f"\n... ({len(result.value)} chars total)"
                        )

                    response = self.llm.send(
                        f"Result of {op.op}:\n{display_value}"
                    )
                except Exception as e:
                    self.trace_collector.record_explore_step(
                        self.trace_node, step_number=explore_steps,
                        elapsed_s=time.monotonic() - step_start,
                        op_type=op.op.value, op_args=op.args, op_bind=op.bind,
                        result_value="", cached=False, error=str(e),
                    )
                    if self.config.verbose:
                        self.console.print(
                            f"[red]EXPLORE step {explore_steps} error ({op.op.value}): "
                            f"{str(e)[:300]}[/red]"
                        )
                    response = self.llm.send(f"Error executing {op.op}: {e}")

            elif isinstance(action, CommitPlan):
                commit_cycles += 1
                if commit_cycles > self.config.max_commit_cycles:
                    commit_overflow_nudges += 1
                    if commit_overflow_nudges >= max_overflow_nudges:
                        raise RuntimeError(
                            f"LLM did not provide a FINAL answer after {max_overflow_nudges} "
                            f"nudges past the {self.config.max_commit_cycles} commit cycle limit."
                        )
                    response = self.llm.send(
                        f"You have reached the maximum of {self.config.max_commit_cycles} "
                        f"commit cycles. Please provide a FINAL answer."
                    )
                    continue

                if self.config.verbose:
                    ops_detail = ", ".join(
                        f"{op.op.value}→{op.bind}" if op.bind else op.op.value
                        for op in action.operations
                    )
                    self.console.print(
                        f"[blue]COMMIT cycle {commit_cycles}/{self.config.max_commit_cycles}: "
                        f"{len(action.operations)} ops [{ops_detail}], "
                        f"output={action.output}[/blue]"
                    )

                try:
                    commit_result, op_traces = self._execute_commit_plan(
                        action, bindings, depth,
                    )
                    bindings[action.output] = commit_result
                    self.trace_collector.record_commit_cycle(
                        self.trace_node, cycle_number=commit_cycles,
                        output_variable=action.output,
                        operations=op_traces, result_value=commit_result,
                    )

                    display_result = commit_result
                    if len(display_result) > self.config.max_result_chars:
                        display_result = (
                            display_result[:self.config.max_result_chars]
                            + f"\n... ({len(commit_result)} chars total)"
                        )

                    response = self.llm.send(
                        f"Commit plan executed. Result ({action.output}):\n{display_result}"
                    )
                except Exception as e:
                    if self.config.verbose:
                        self.console.print(
                            f"[red]COMMIT cycle {commit_cycles} error: {str(e)[:300]}[/red]"
                        )
                    response = self.llm.send(f"Error executing commit plan: {e}")

    def _execute_commit_plan(
        self, plan: CommitPlan, bindings: dict[str, str], depth: int
    ) -> tuple[str, list[CommitOperationTrace]]:
        """Execute a commit plan, handling recursive calls and parallelism."""
        local_bindings = dict(bindings)
        op_traces: list[CommitOperationTrace] = []
        try:
            return self._run_plan_ops(plan, local_bindings, op_traces, depth)
        except Exception as e:
            # Keep what succeeded: a failed final aggregation step must not
            # throw away an expensive map fan-out and make the model redo it.
            preserved = sorted(k for k in local_bindings if k not in bindings)
            bindings.update({k: local_bindings[k] for k in preserved})
            note = (
                f" Results of the operations that succeeded are preserved as "
                f"variables: {', '.join(preserved)}. Reuse them (e.g. fix only "
                f"the failing step with a single explore op) instead of "
                f"re-running the whole plan."
                if preserved else ""
            )
            raise RuntimeError(f"{e}{note}") from e

    def _run_plan_ops(
        self,
        plan: CommitPlan,
        local_bindings: dict[str, str],
        op_traces: list[CommitOperationTrace],
        depth: int,
    ) -> tuple[str, list[CommitOperationTrace]]:
        ops = plan.operations
        i = 0
        while i < len(ops):
            op = ops[i]
            step_start = time.monotonic()
            child_trace_ids: list[int] = []

            if op.op == OpType.RLM_CALL:
                # Run a maximal run of consecutive rlm_call ops whose contexts
                # are already bound (i.e. independent of each other) in
                # parallel, instead of one slow leaf call after another.
                group = [op]
                produced = {op.bind} if op.bind else set()
                j = i + 1
                while j < len(ops) and ops[j].op == OpType.RLM_CALL:
                    nxt = ops[j]
                    if nxt.args.get("context") in produced:
                        break
                    group.append(nxt)
                    if nxt.bind:
                        produced.add(nxt.bind)
                    j += 1
                group_results = self._parallel_rlm_calls(group, local_bindings, depth)
                for k, (grp_op, (result_value, child)) in enumerate(zip(group, group_results, strict=True)):
                    idx = i + k + 1
                    step_elapsed = time.monotonic() - step_start
                    if self.config.verbose:
                        op_desc = self._format_op(grp_op)
                        bind_note = f" → {grp_op.bind}" if grp_op.bind else ""
                        par_note = f", parallel x{len(group)}" if len(group) > 1 else ""
                        self.console.print(
                            f"[dim]  {idx}. {op_desc}{bind_note}  ({step_elapsed:.3f}s{par_note})[/dim]"
                        )
                    if grp_op.bind:
                        local_bindings[grp_op.bind] = result_value
                    if self.trace_collector.enabled:
                        op_traces.append(CommitOperationTrace(
                            index=idx,
                            operation_op=grp_op.op.value,
                            operation_args=grp_op.args,
                            operation_bind=grp_op.bind,
                            elapsed_s=step_elapsed,
                            result_value=result_value,
                            child_trace_ids=[child.trace_node.trace_id],
                        ))
                i = j
                continue

            if op.op == OpType.MAP:
                prompt = op.args["prompt"]
                input_ref = op.args["input"]
                raw = local_bindings[input_ref]
                items: list[str] = parse_list_value(raw)
                before_count = len(self.child_orchestrators)
                result_value = self._parallel_map(prompt, items, depth)
                if self.trace_collector.enabled:
                    for child in self.child_orchestrators[before_count:]:
                        child_trace_ids.append(child.trace_node.trace_id)

            else:
                result = self.evaluator.execute(op, local_bindings)
                result_value = result.value

            step_elapsed = time.monotonic() - step_start
            idx = i + 1

            if self.config.verbose:
                op_desc = self._format_op(op)
                bind_note = f" → {op.bind}" if op.bind else ""
                self.console.print(
                    f"[dim]  {idx}. {op_desc}{bind_note}  ({step_elapsed:.3f}s)[/dim]"
                )

            if op.bind:
                local_bindings[op.bind] = result_value

            if self.trace_collector.enabled:
                op_traces.append(CommitOperationTrace(
                    index=idx,
                    operation_op=op.op.value,
                    operation_args=op.args,
                    operation_bind=op.bind,
                    elapsed_s=step_elapsed,
                    result_value=result_value,
                    child_trace_ids=child_trace_ids,
                ))
            i += 1

        if plan.output not in local_bindings:
            raise KeyError(
                f"Plan output variable {plan.output!r} was never bound; "
                f"available: {', '.join(sorted(local_bindings))}"
            )
        return local_bindings[plan.output], op_traces

    def _parallel_rlm_calls(
        self, group: list[Operation], bindings: dict[str, str], depth: int
    ) -> list[tuple[str, RLMOrchestrator]]:
        """Execute independent rlm_call ops concurrently, preserving order."""
        contexts = [bindings[op.args["context"]] for op in group]
        if len(group) == 1:
            return [self._spawn_child(group[0].args["query"], contexts[0], depth)]
        results: list[tuple[str, RLMOrchestrator] | None] = [None] * len(group)
        with ThreadPoolExecutor(max_workers=self.config.max_parallel_jobs) as executor:
            futures = {
                executor.submit(self._spawn_child, op.args["query"], ctx, depth): n
                for n, (op, ctx) in enumerate(zip(group, contexts, strict=True))
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return [r for r in results if r is not None]

    def _recursive_call(self, query: str, context_text: str, depth: int) -> str:
        """Spawn a recursive RLM call and return its answer."""
        return self._spawn_child(query, context_text, depth)[0]

    def _spawn_child(
        self, query: str, context_text: str, depth: int
    ) -> tuple[str, RLMOrchestrator]:
        """Spawn a child orchestrator; return (answer, child) for tracing."""
        with self.profile.measure("recursive", "recursive_call", depth=depth + 1):
            child_config = self.config
            if self.config.child_model:
                child_config = self.config.model_copy(update={
                    "model": self.config.child_model,
                    "child_model": None,
                })
            sub_orchestrator = RLMOrchestrator(
                child_config, parent=self,
                trace_collector=self.trace_collector,
            )
            self.child_orchestrators.append(sub_orchestrator)
            result = sub_orchestrator.run(query, context_text, depth=depth + 1)
            if self.trace_collector.enabled:
                self.trace_node.children.append(sub_orchestrator.trace_node)
            return result, sub_orchestrator

    def _parallel_map(self, prompt: str, items: list[str], depth: int) -> str:
        """Execute map operation with parallel recursive calls."""
        with self.profile.measure("parallel", "parallel_map", item_count=len(items)):
            results = [""] * len(items)
            # Spawning the piece at max depth makes the child a direct call
            # while keeping child_model, tracing and token accounting intact.
            leaf_depth = self.config.max_recursion_depth if self.config.map_direct else depth

            with ThreadPoolExecutor(max_workers=self.config.max_parallel_jobs) as executor:
                futures = {
                    executor.submit(self._recursive_call, prompt, item, leaf_depth): i
                    for i, item in enumerate(items)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    results[idx] = future.result()

            return json.dumps(results)

    def _direct_call(self, query: str, context_text: str) -> str:
        """Direct LLM call at max recursion depth (no explore/commit)."""
        max_chars = 100_000
        truncated = context_text[:max_chars]
        system_prompt = (
            "Answer the following query based on the provided context. "
            "Be precise and concise."
        )

        # Content-addressed cache for leaf calls: a repeated map/rlm_call over
        # the same query and context (models often re-run an identical plan)
        # then costs nothing instead of another full fan-out of LLM calls.
        key_data = {
            "kind": "direct_call",
            "model": self.config.model,
            "temperature": self.config.temperature,
            "reasoning_strength": self.config.reasoning_strength,
            "max_output_tokens": self.config.max_output_tokens,
            "system": system_prompt,
            "query": query,
            "context": truncated,
        }
        cache_key = hashlib.sha256(
            json.dumps(key_data, sort_keys=True).encode()
        ).hexdigest()
        cached = self.cache.get(cache_key)
        if cached is not None:
            self.profile.record_cache_hit()
            return cached
        self.profile.record_cache_miss()

        client = LLMClient(self.config, verbose=self.config.verbose,
                            console=self.console,
                            trace=self.trace_collector,
                            trace_node=self.trace_node,
                            use_tools=False)
        client.set_system_prompt(system_prompt)
        try:
            answer = client.send(f"Query: {query}\n\nContext:\n{truncated}")
            self.cache.put(cache_key, answer)
            return answer
        finally:
            # Roll the throwaway client's usage into this orchestrator's totals
            # so direct calls are counted in token usage and cost reports.
            direct_in, direct_out = client.get_token_usage()
            self.llm.total_input_tokens += direct_in
            self.llm.total_output_tokens += direct_out

    def _format_op(self, op: Operation) -> str:
        """Format an operation for human-readable display."""
        parts = []
        for k, v in op.args.items():
            if isinstance(v, str) and len(v) > 40:
                parts.append(f'{k}="{v[:37]}..."')
            elif isinstance(v, str):
                parts.append(f'{k}="{v}"')
            else:
                parts.append(f"{k}={v}")
        return f"{op.op.value}({', '.join(parts)})"

    def get_total_token_usage(self) -> tuple[int, int]:
        """Get total token usage including all child orchestrators."""
        input_tokens, output_tokens = self.llm.get_token_usage()
        for child in self.child_orchestrators:
            child_input, child_output = child.get_total_token_usage()
            input_tokens += child_input
            output_tokens += child_output
        return input_tokens, output_tokens

    def get_total_cost(self, pricing_fn: Callable[[str, int, int], float]) -> float:
        """Get total cost including all child orchestrators, priced per-model."""
        input_tokens, output_tokens = self.llm.get_token_usage()
        cost = pricing_fn(self.config.model, input_tokens, output_tokens)
        for child in self.child_orchestrators:
            cost += child.get_total_cost(pricing_fn)
        return cost

    def get_total_profile(self) -> TimingProfile:
        """Get merged timing profile including all child orchestrators."""
        merged = TimingProfile(enabled=self.profile.enabled)
        merged.merge(self.profile)
        for child in self.child_orchestrators:
            merged.merge(child.get_total_profile())
        return merged

    def get_trace(self) -> ExecutionTrace:
        """Return the execution trace for this orchestrator."""
        from datetime import datetime, timezone
        return ExecutionTrace(
            timestamp=datetime.now(timezone.utc).isoformat(),
            root=self.trace_node,
        )
