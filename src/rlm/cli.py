"""CLI entry point for rlm-secure."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import click
from rich.console import Console

from rlm.config import load_config

console = Console(stderr=True)


# Pricing per million tokens (as of Feb 2026)
MODEL_PRICING = {
    # Anthropic Claude models (https://platform.claude.com/docs/en/about-claude/pricing)
    "claude-opus-4.6": {"input": 5.00, "output": 25.00},
    "claude-opus-4.5": {"input": 5.00, "output": 25.00},
    "claude-opus-4.1": {"input": 15.00, "output": 75.00},
    "claude-opus-4": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4.5": {"input": 3.00, "output": 15.00},
    "claude-haiku-4.5": {"input": 1.00, "output": 5.00},
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},   # API ID alias
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},   # API ID alias
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},   # Claude Haiku 3.5
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00}, # Claude Sonnet 3.5

    # OpenAI GPT text models, Standard tier (https://platform.openai.com/docs/pricing)
    "gpt-5.2": {"input": 1.75, "output": 14.00},
    "gpt-5.1": {"input": 1.25, "output": 10.00},
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in dollars based on model and token counts."""
    pricing = MODEL_PRICING.get(model, {"input": 1.00, "output": 5.00})
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


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
@click.option("--model", "-m", default=None, help="LLM model to use.")
@click.option("--max-explore", default=None, type=int, help="Max explore steps.")
@click.option("--max-depth", default=None, type=int, help="Max recursion depth.")
@click.option("--use-nix", is_flag=True, default=False, help="Use Nix for sandboxing.")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Verbose output.")
@click.option("--trace", "trace_path", type=click.Path(), default=None,
              help="Write execution trace JSON to PATH.")
def run(
    query: str,
    context: str | None,
    model: str | None,
    max_explore: int | None,
    max_depth: int | None,
    use_nix: bool,
    verbose: bool,
    trace_path: str | None,
) -> None:
    """Run an RLM query against a context."""
    config = load_config(
        model=model,
        max_explore_steps=max_explore,
        max_recursion_depth=max_depth,
        use_nix=use_nix,
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
        console.print(f"[dim]Context: {len(context_text):,} chars[/dim]")
        if config.use_nix:
            console.print("[dim]Nix sandboxing: enabled[/dim]")

    from rlm.orchestrator import RLMOrchestrator
    from rlm.trace import TraceCollector

    trace_collector = TraceCollector(enabled=trace_path is not None)

    start = time.monotonic()
    orchestrator = RLMOrchestrator(config, trace_collector=trace_collector)
    answer = orchestrator.run(query, context_text)
    elapsed = time.monotonic() - start

    click.echo(answer)

    if verbose:
        input_tokens, output_tokens = orchestrator.get_total_token_usage()
        total_tokens = input_tokens + output_tokens
        cost = estimate_cost(config.model, input_tokens, output_tokens)

        console.print(f"\n[dim]Completed in {elapsed:.1f}s[/dim]")
        console.print(f"[dim]Tokens: {input_tokens:,} in + {output_tokens:,} out = {total_tokens:,} total[/dim]")
        console.print(f"[dim]Estimated cost: ${cost:.4f}[/dim]")

        from rlm.timing import TimingProfile
        merged_profile = orchestrator.get_total_profile()
        TimingProfile.print_summary(merged_profile, elapsed, console)

    if trace_path is not None:
        trace = orchestrator.get_trace()
        trace_file = Path(trace_path)
        TraceCollector.write_trace(trace, trace_file)
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
