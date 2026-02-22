"""Results reporting for OOLONG benchmark evaluation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

from rlm.eval.runner import EvalResult


def load_results(path: Path) -> list[EvalResult]:
    """Load EvalResult objects from a JSONL file."""
    results: list[EvalResult] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(EvalResult.model_validate_json(line))
    return results


def print_report(results: list[EvalResult], console: Console | None = None) -> None:
    """Print a Rich summary report of evaluation results.

    Args:
        results: List of evaluation results.
        console: Optional Rich console (defaults to stderr).
    """
    if console is None:
        console = Console(stderr=True)

    if not results:
        console.print("[yellow]No results to report.[/yellow]")
        return

    # Overall metrics
    total = len(results)
    overall_score = sum(r.score for r in results) / total
    total_cost = sum(r.cost_usd for r in results)
    total_input = sum(r.input_tokens for r in results)
    total_output = sum(r.output_tokens for r in results)
    mean_elapsed = sum(r.elapsed_s for r in results) / total
    errors = sum(1 for r in results if r.error is not None)

    # Summary table
    summary = Table(title="OOLONG Evaluation Summary")
    summary.add_column("Metric", style="bold")
    summary.add_column("Value")

    summary.add_row("Tasks", str(total))
    summary.add_row("Overall Score", f"{overall_score:.3f} ({overall_score * 100:.1f}%)")
    summary.add_row("Errors", str(errors))
    summary.add_row("Total Cost", f"${total_cost:.4f}")
    summary.add_row("Total Tokens", f"{total_input + total_output:,} ({total_input:,} in + {total_output:,} out)")
    summary.add_row("Mean Time/Task", f"{mean_elapsed:.1f}s")

    console.print(summary)
    console.print()

    # Score by task_group
    group_scores: dict[str, list[float]] = defaultdict(list)
    for r in results:
        group_scores[r.task_group].append(r.score)

    group_table = Table(title="Score by Task Group")
    group_table.add_column("Task Group", style="bold")
    group_table.add_column("Count")
    group_table.add_column("Score")

    for group in sorted(group_scores.keys()):
        scores = group_scores[group]
        avg = sum(scores) / len(scores)
        group_table.add_row(group, str(len(scores)), f"{avg:.3f} ({avg * 100:.1f}%)")

    console.print(group_table)
    console.print()

    # Score by answer_type
    type_scores: dict[str, list[float]] = defaultdict(list)
    for r in results:
        type_scores[r.answer_type].append(r.score)

    type_table = Table(title="Score by Answer Type")
    type_table.add_column("Answer Type", style="bold")
    type_table.add_column("Count")
    type_table.add_column("Score")

    for atype in sorted(type_scores.keys()):
        scores = type_scores[atype]
        avg = sum(scores) / len(scores)
        type_table.add_row(atype, str(len(scores)), f"{avg:.3f} ({avg * 100:.1f}%)")

    console.print(type_table)
    console.print()

    # Paper comparison
    comparison = Table(title="Comparison with RLM Paper (Zhang & Khattab, 2025)")
    comparison.add_column("Method", style="bold")
    comparison.add_column("Score")
    comparison.add_column("Notes")

    comparison.add_row(
        "Paper baseline (GPT-5)", "44.0%",
        "Direct prompting, trec_coarse 131K",
    )
    comparison.add_row(
        "Paper RLM (GPT-5 + GPT-5-mini)", "56.5%",
        "RLM, trec_coarse 131K",
    )
    comparison.add_row(
        "This run", f"{overall_score * 100:.1f}%",
        f"{total} tasks",
    )

    console.print(comparison)
