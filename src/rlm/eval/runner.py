"""Benchmark runner with JSONL checkpointing."""

from __future__ import annotations

import json
import time
from pathlib import Path

from pydantic import BaseModel
from rich.console import Console

from rlm.eval.datasets import EvalTask
from rlm.eval.scoring import parse_oolong_answer, score_oolong_synth
from rlm.pricing import estimate_cost
from rlm.types import RLMConfig


class EvalResult(BaseModel):
    """Result of evaluating a single task."""
    id: int
    context_window_id: int
    question: str
    predicted: str
    gold: str
    score: float
    answer_type: str
    task_group: str
    cost_usd: float
    input_tokens: int
    output_tokens: int
    elapsed_s: float
    error: str | None = None


class EvalRunner:
    """Runs benchmark tasks with resumable JSONL checkpointing."""

    def __init__(self, config: RLMConfig, output_path: Path) -> None:
        # Enable benchmark-friendly eval prompt when Wasm sandbox is available
        if config.wasm_python_path and not config.benchmark_eval_prompt:
            config = config.model_copy(update={"benchmark_eval_prompt": True})
        self.config = config
        self.output_path = output_path
        self.console = Console(stderr=True)

    def load_completed(self) -> set[int]:
        """Read existing JSONL and return set of completed task IDs."""
        completed: set[int] = set()
        if not self.output_path.exists():
            return completed
        with open(self.output_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    completed.add(obj["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        return completed

    def run_task(self, task: EvalTask) -> EvalResult:
        """Run a single evaluation task through the RLM orchestrator."""
        from rlm.orchestrator import RLMOrchestrator
        from rlm.trace import TraceCollector

        trace_collector = TraceCollector(enabled=False)
        orchestrator = RLMOrchestrator(self.config, trace_collector=trace_collector)

        start = time.monotonic()
        error: str | None = None
        predicted_raw = ""

        try:
            predicted_raw = orchestrator.run(task.question, task.context_text)
        except Exception as e:
            error = str(e)

        elapsed = time.monotonic() - start
        input_tokens, output_tokens = orchestrator.get_total_token_usage()
        cost = orchestrator.get_total_cost(estimate_cost)

        predicted = parse_oolong_answer(predicted_raw)
        gold = parse_oolong_answer(task.answer)
        score = score_oolong_synth(predicted, gold, task.answer_type) if not error else 0.0

        return EvalResult(
            id=task.id,
            context_window_id=task.context_window_id,
            question=task.question,
            predicted=predicted,
            gold=gold,
            score=score,
            answer_type=task.answer_type,
            task_group=task.task_group,
            cost_usd=cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            elapsed_s=elapsed,
            error=error,
        )

    def run_all(
        self,
        tasks: list[EvalTask],
        resume: bool = False,
        limit: int | None = None,
    ) -> list[EvalResult]:
        """Run all tasks, appending results incrementally to JSONL.

        Args:
            tasks: List of evaluation tasks to run.
            resume: If True, skip tasks already in the output JSONL.
            limit: Optional limit on number of tasks to run.

        Returns:
            List of all EvalResult objects (including previously completed if resuming).
        """
        completed_ids = self.load_completed() if resume else set()
        if completed_ids:
            self.console.print(f"[dim]Resuming: {len(completed_ids)} tasks already completed[/dim]")

        # Filter to pending tasks
        pending = [t for t in tasks if t.id not in completed_ids]
        if limit is not None:
            pending = pending[:limit]

        total = len(pending)
        if total == 0:
            self.console.print("[green]All tasks already completed.[/green]")
            return self._load_all_results()

        self.console.print(f"Running {total} tasks ({len(completed_ids)} already done)")

        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        results: list[EvalResult] = []
        running_score = 0.0
        running_cost = 0.0

        for i, task in enumerate(pending, 1):
            self.console.print(
                f"\n[bold]Task {i}/{total}[/bold] (id={task.id}, "
                f"type={task.answer_type}, group={task.task_group})"
            )

            result = self.run_task(task)
            results.append(result)

            # Append to JSONL immediately
            with open(self.output_path, "a") as f:
                f.write(result.model_dump_json() + "\n")

            running_score = (running_score * (i - 1) + result.score) / i
            running_cost += result.cost_usd

            status = "[green]OK[/green]" if result.error is None else f"[red]ERR: {result.error}[/red]"
            self.console.print(
                f"  {status} | score={result.score:.2f} | "
                f"predicted={result.predicted!r} | gold={result.gold!r} | "
                f"{result.elapsed_s:.1f}s | ${result.cost_usd:.4f}"
            )
            self.console.print(
                f"  [dim]Running avg: {running_score:.3f} | Total cost: ${running_cost:.4f}[/dim]"
            )

        return results

    def _load_all_results(self) -> list[EvalResult]:
        """Load all results from the JSONL file."""
        results: list[EvalResult] = []
        if not self.output_path.exists():
            return results
        with open(self.output_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                results.append(EvalResult.model_validate_json(line))
        return results
