"""RLM Orchestrator — manages the explore/commit protocol loop."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console

from rlm.cache.store import CacheStore
from rlm.evaluator.lightweight import LightweightEvaluator
from rlm.llm.client import LLMClient
from rlm.llm.parser import ParseError, parse_llm_output
from rlm.llm.prompts import SYSTEM_PROMPT
from rlm.types import (
    CommitPlan,
    Context,
    ExploreAction,
    FinalAnswer,
    OpType,
    RLMConfig,
)

logger = logging.getLogger(__name__)


class RLMOrchestrator:
    """Orchestrates the explore/commit protocol between the LLM and evaluators."""

    def __init__(self, config: RLMConfig):
        self.config = config
        self.llm = LLMClient(config)
        self.cache = CacheStore(config.cache_dir)
        self.evaluator = LightweightEvaluator(cache=self.cache)
        self.console = Console(stderr=True)

        if config.use_nix:
            from rlm.nix.builder import NixBuilder
            self.nix_builder = NixBuilder(max_jobs=config.max_parallel_jobs)
            if not self.nix_builder.available:
                raise RuntimeError(
                    "Nix is not installed but --use-nix was specified. "
                    "Install Nix from https://nixos.org/ or remove --use-nix."
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
        if depth > self.config.max_recursion_depth:
            return self._direct_call(query, context_text)

        ctx = Context(content=context_text)
        bindings: dict[str, str] = {"context": ctx.content}

        system = SYSTEM_PROMPT.format(
            context_chars=f"{len(ctx.content):,}",
            query=query,
        )
        self.llm.set_system_prompt(system)

        explore_steps = 0
        commit_cycles = 0

        response = self.llm.send("Begin. The context variable is available.")

        while True:
            try:
                action = parse_llm_output(response)
            except ParseError as e:
                logger.warning("Parse error: %s", e)
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
                return action.answer

            elif isinstance(action, ExploreAction):
                explore_steps += 1
                if explore_steps > self.config.max_explore_steps:
                    response = self.llm.send(
                        f"You have reached the maximum of {self.config.max_explore_steps} "
                        f"explore steps. Please COMMIT a plan or provide a FINAL answer."
                    )
                    continue

                if self.config.verbose:
                    self.console.print(
                        f"[dim]EXPLORE [{explore_steps}]: "
                        f"{action.operation.op}({action.operation.args})[/dim]"
                    )

                try:
                    result = self.evaluator.execute(action.operation, bindings)
                    if action.operation.bind:
                        bindings[action.operation.bind] = result.value

                    display_value = result.value
                    if len(display_value) > 4000:
                        display_value = (
                            display_value[:4000]
                            + f"\n... ({len(result.value)} chars total)"
                        )

                    response = self.llm.send(
                        f"Result of {action.operation.op}:\n{display_value}"
                    )
                except Exception as e:
                    response = self.llm.send(f"Error executing {action.operation.op}: {e}")

            elif isinstance(action, CommitPlan):
                commit_cycles += 1
                if commit_cycles > self.config.max_commit_cycles:
                    response = self.llm.send(
                        f"You have reached the maximum of {self.config.max_commit_cycles} "
                        f"commit cycles. Please provide a FINAL answer."
                    )
                    continue

                if self.config.verbose:
                    self.console.print(
                        f"[blue]COMMIT [{commit_cycles}]: "
                        f"{len(action.operations)} operations[/blue]"
                    )

                try:
                    commit_result = self._execute_commit_plan(action, bindings, depth)
                    bindings[action.output] = commit_result

                    display_result = commit_result
                    if len(display_result) > 4000:
                        display_result = (
                            display_result[:4000]
                            + f"\n... ({len(commit_result)} chars total)"
                        )

                    response = self.llm.send(
                        f"Commit plan executed. Result ({action.output}):\n{display_result}"
                    )
                except Exception as e:
                    response = self.llm.send(f"Error executing commit plan: {e}")

    def _execute_commit_plan(
        self, plan: CommitPlan, bindings: dict[str, str], depth: int
    ) -> str:
        """Execute a commit plan, handling recursive calls and parallelism."""
        local_bindings = dict(bindings)

        for op in plan.operations:
            if op.op == OpType.RLM_CALL:
                query = op.args["query"]
                ctx_ref = op.args["context"]
                ctx_text = local_bindings[ctx_ref]
                result_value = self._recursive_call(query, ctx_text, depth)

            elif op.op == OpType.MAP:
                prompt = op.args["prompt"]
                input_ref = op.args["input"]
                raw = local_bindings[input_ref]
                items: list[str] = json.loads(raw) if raw.startswith("[") else [raw]
                result_value = self._parallel_map(prompt, items, depth)

            else:
                result = self.evaluator.execute(op, local_bindings)
                result_value = result.value

            if op.bind:
                local_bindings[op.bind] = result_value

        return local_bindings[plan.output]

    def _recursive_call(self, query: str, context_text: str, depth: int) -> str:
        """Spawn a recursive RLM call."""
        sub_orchestrator = RLMOrchestrator(self.config)
        return sub_orchestrator.run(query, context_text, depth=depth + 1)

    def _parallel_map(self, prompt: str, items: list[str], depth: int) -> str:
        """Execute map operation with parallel recursive calls."""
        results = [""] * len(items)

        with ThreadPoolExecutor(max_workers=self.config.max_parallel_jobs) as executor:
            futures = {
                executor.submit(self._recursive_call, prompt, item, depth): i
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

        client = LLMClient(self.config)
        client.set_system_prompt(
            "Answer the following query based on the provided context. "
            "Be precise and concise."
        )
        return client.send(f"Query: {query}\n\nContext:\n{truncated}")
