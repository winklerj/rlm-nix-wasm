"""CLI entry point for rlm-nix-wasm."""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import click
from rich.console import Console

from rlm.config import load_config
from rlm.pricing import MODEL_PRICING, estimate_cost

console = Console(stderr=True)


@click.group()
@click.version_option(version="0.1.0")
def main() -> None:
    """Sandboxed Recursive Language Models with Nix."""
    pass


@main.command("list-model-pricing")
def list_model_pricing() -> None:
    """List known model pricing."""
    try:
        from rich.table import Table
    except ImportError:  # pragma: no cover - optional pretty output
        for name, pricing in sorted(MODEL_PRICING.items()):
            click.echo(f"{name}: input=${pricing['input']}/M, output=${pricing['output']}/M")
        return

    table = Table(title="Model pricing (per 1M tokens)")
    table.add_column("Model")
    table.add_column("Input ($/M)")
    table.add_column("Output ($/M)")

    for name, pricing in sorted(MODEL_PRICING.items()):
        table.add_row(
            name,
            f"{pricing['input']:.3f}",
            f"{pricing['output']:.3f}",
        )

    console.print(table)


@main.command()
@click.option("--query", "-q", required=True, help="The query to answer.")
@click.option("--context", "-c", type=click.Path(exists=True), help="Path to context file.")
@click.option("--model", "-m", default=None, help="LLM model for the orchestrator.")
@click.option("--child-model", default=None, help="LLM model for recursive sub-calls (defaults to --model).")
@click.option("--max-explore", default=None, type=int, help="Max explore steps.")
@click.option("--max-depth", default=None, type=int, help="Max recursion depth.")
@click.option("--no-nix", is_flag=True, default=False, help="Disable Nix sandboxing.")
@click.option("--wasm-python", type=click.Path(), default=None,
              help="Path to python.wasm for sandboxed code execution.")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Verbose output.")
@click.option("--trace", is_flag=True, default=False,
              help="Write execution trace JSON to traces/ directory.")
def run(
    query: str,
    context: str | None,
    model: str | None,
    child_model: str | None,
    max_explore: int | None,
    max_depth: int | None,
    no_nix: bool,
    wasm_python: str | None,
    verbose: bool,
    trace: bool,
) -> None:
    """Run an RLM query against a context."""
    config = load_config(
        model=model,
        child_model=child_model,
        max_explore_steps=max_explore,
        max_recursion_depth=max_depth,
        use_nix=not no_nix,
        wasm_python_path=Path(wasm_python) if wasm_python else None,
        verbose=verbose,
    )

    # Read context from file or stdin
    if context:
        context_text = Path(context).read_text()
    elif not sys.stdin.isatty():
        context_text = sys.stdin.read()
    else:
        console.print("[red]Error: provide --context or pipe context via stdin[/red]")
        raise SystemExit(1)

    if verbose:
        console.print(f"[dim]Model: {config.model}[/dim]")
        if config.child_model:
            console.print(f"[dim]Child model: {config.child_model}[/dim]")
        console.print(f"[dim]Context: {len(context_text):,} chars[/dim]")
        console.print(f"[dim]Nix sandboxing: {'enabled' if config.use_nix else 'disabled'}[/dim]")
        if config.wasm_python_path:
            console.print(f"[dim]Wasm sandbox: {config.wasm_python_path}[/dim]")

    from rlm.orchestrator import RLMOrchestrator
    from rlm.trace import TraceCollector

    trace_collector = TraceCollector(enabled=trace)

    start = time.monotonic()
    orchestrator = RLMOrchestrator(config, trace_collector=trace_collector)
    answer = orchestrator.run(query, context_text)
    elapsed = time.monotonic() - start

    click.echo(answer)

    if verbose:
        input_tokens, output_tokens = orchestrator.get_total_token_usage()
        total_tokens = input_tokens + output_tokens
        cost = orchestrator.get_total_cost(estimate_cost)

        console.print(f"\n[dim]Completed in {elapsed:.1f}s[/dim]")
        console.print(f"[dim]Tokens: {input_tokens:,} in + {output_tokens:,} out = {total_tokens:,} total[/dim]")
        console.print(f"[dim]Estimated cost: ${cost:.4f}[/dim]")

        from rlm.timing import TimingProfile
        merged_profile = orchestrator.get_total_profile()
        TimingProfile.print_summary(merged_profile, elapsed, console)

    if trace:
        execution_trace = orchestrator.get_trace()
        now = datetime.now(timezone.utc)
        trace_dir = Path("traces")
        trace_dir.mkdir(exist_ok=True)
        trace_file = trace_dir / f"{now.strftime('%Y-%m-%dT%H-%M-%S')}.{now.strftime('%f')[:3]}.json"
        TraceCollector.write_trace(execution_trace, trace_file)
        console.print(f"[dim]Trace written to {trace_file}[/dim]")


@main.group()
def cache() -> None:
    """Cache management commands."""
    pass


@cache.command()
def stats() -> None:
    """Show cache statistics."""
    from rlm.cache.store import CacheStore

    config = load_config()
    store = CacheStore(config.cache_dir)
    s = store.stats()
    click.echo(f"Cache dir: {s['cache_dir']}")
    click.echo(f"Entries:   {s['entries']}")
    click.echo(f"Size:      {s['size_human']}")


@cache.command()
def clear() -> None:
    """Clear the operation cache."""
    from rlm.cache.store import CacheStore

    config = load_config()
    store = CacheStore(config.cache_dir)
    removed = store.clear()
    click.echo(f"Cleared {removed} entries from {config.cache_dir}")


# --- Eval subcommands ---


@main.group("eval")
def eval_group() -> None:
    """Benchmark evaluation commands."""
    pass


@eval_group.command("download")
@click.option("--benchmark", default="oolong-synth",
              help="Benchmark to download (default: oolong-synth).")
def eval_download(benchmark: str) -> None:
    """Download benchmark dataset from HuggingFace."""
    if benchmark != "oolong-synth":
        console.print(f"[red]Unknown benchmark: {benchmark}. Only 'oolong-synth' is supported.[/red]")
        raise SystemExit(1)

    from rlm.eval.datasets import download_oolong_synth

    console.print(f"Downloading {benchmark}...")
    path = download_oolong_synth()
    console.print(f"[green]Dataset cached at {path}[/green]")


@eval_group.command("run")
@click.option("--benchmark", default="oolong-synth", help="Benchmark name.")
@click.option("--dataset", "dataset_name", default="trec_coarse",
              help="Dataset within the benchmark (default: trec_coarse).")
@click.option("--context-len", default=65536, type=int,
              help="Context window length to evaluate (default: 65536).")
@click.option("--model", "-m", default=None, help="LLM model for the orchestrator.")
@click.option("--child-model", default=None,
              help="LLM model for recursive sub-calls (defaults to --model).")
@click.option("--max-explore", default=None, type=int, help="Max explore steps.")
@click.option("--max-depth", default=None, type=int, help="Max recursion depth.")
@click.option("--no-nix", is_flag=True, default=False, help="Disable Nix sandboxing.")
@click.option("--wasm-python", type=click.Path(), default=None,
              help="Path to python.wasm for sandboxed code execution.")
@click.option("--temperature", default=0.3, type=float,
              help="LLM temperature (default: 0.3 for eval).")
@click.option("--output", "-o", default="results/oolong.jsonl", type=click.Path(),
              help="Output JSONL path (default: results/oolong.jsonl).")
@click.option("--resume/--no-resume", default=False,
              help="Resume from existing results, skipping completed tasks.")
@click.option("--limit", default=None, type=int,
              help="Only run first N tasks (for testing).")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Verbose output.")
@click.option("--trace", is_flag=True, default=False,
              help="Enable execution tracing.")
def eval_run(
    benchmark: str,
    dataset_name: str,
    context_len: int,
    model: str | None,
    child_model: str | None,
    max_explore: int | None,
    max_depth: int | None,
    no_nix: bool,
    wasm_python: str | None,
    temperature: float,
    output: str,
    resume: bool,
    limit: int | None,
    verbose: bool,
    trace: bool,
) -> None:
    """Run benchmark evaluation."""
    if benchmark != "oolong-synth":
        console.print(f"[red]Unknown benchmark: {benchmark}[/red]")
        raise SystemExit(1)

    from rlm.eval.datasets import load_oolong_synth_tasks
    from rlm.eval.report import print_report
    from rlm.eval.runner import EvalRunner

    config = load_config(
        model=model,
        child_model=child_model,
        max_explore_steps=max_explore,
        max_recursion_depth=max_depth,
        use_nix=not no_nix,
        wasm_python_path=Path(wasm_python) if wasm_python else None,
        temperature=temperature,
        verbose=verbose,
    )

    console.print(f"[bold]OOLONG Evaluation: {dataset_name} @ {context_len:,} tokens[/bold]")
    console.print(f"[dim]Model: {config.model}[/dim]")
    if config.child_model:
        console.print(f"[dim]Child model: {config.child_model}[/dim]")

    # trec_coarse lives in the validation split
    split = "validation" if dataset_name == "trec_coarse" else "test"
    console.print(f"Loading tasks (split={split})...")
    tasks = load_oolong_synth_tasks(
        dataset_name=dataset_name,
        context_len=context_len,
        split=split,
    )

    if not tasks:
        console.print(
            f"[red]No tasks found for dataset={dataset_name}, "
            f"context_len={context_len}, split={split}[/red]"
        )
        raise SystemExit(1)

    console.print(f"Found {len(tasks)} tasks")

    runner = EvalRunner(config, output_path=Path(output))
    results = runner.run_all(tasks, resume=resume, limit=limit)

    if results:
        console.print()
        print_report(results, console=console)


@eval_group.command("report")
@click.argument("results_path", type=click.Path(exists=True))
def eval_report(results_path: str) -> None:
    """Show summary report from a results JSONL file."""
    from rlm.eval.report import load_results, print_report

    results = load_results(Path(results_path))
    if not results:
        console.print("[yellow]No results found in file.[/yellow]")
        raise SystemExit(1)

    print_report(results, console=console)
