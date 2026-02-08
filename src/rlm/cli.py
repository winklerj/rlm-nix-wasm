"""CLI entry point for rlm-secure."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import click
from rich.console import Console

from rlm.config import load_config

console = Console(stderr=True)


@click.group()
@click.version_option(version="0.1.0")
def main() -> None:
    """Sandboxed Recursive Language Models with Nix."""
    pass


@main.command()
@click.option("--query", "-q", required=True, help="The query to answer.")
@click.option("--context", "-c", type=click.Path(exists=True), help="Path to context file.")
@click.option("--model", "-m", default=None, help="LLM model to use.")
@click.option("--max-explore", default=None, type=int, help="Max explore steps.")
@click.option("--max-depth", default=None, type=int, help="Max recursion depth.")
@click.option("--use-nix", is_flag=True, default=False, help="Use Nix for sandboxing.")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Verbose output.")
def run(
    query: str,
    context: str | None,
    model: str | None,
    max_explore: int | None,
    max_depth: int | None,
    use_nix: bool,
    verbose: bool,
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

    start = time.monotonic()
    orchestrator = RLMOrchestrator(config)
    answer = orchestrator.run(query, context_text)
    elapsed = time.monotonic() - start

    click.echo(answer)

    if verbose:
        console.print(f"\n[dim]Completed in {elapsed:.1f}s[/dim]")


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
