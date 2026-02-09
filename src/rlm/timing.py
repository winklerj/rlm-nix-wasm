"""Lightweight timing instrumentation for RLM operations."""

from __future__ import annotations

import statistics
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator

from pydantic import BaseModel
from rich.console import Console
from rich.table import Table


class TimingEntry(BaseModel):
    """A single timing measurement."""
    category: str       # "llm", "evaluator", "cache", "parse", "recursive", "parallel"
    label: str          # "send", "grep", "lookup", "parse_response", etc.
    elapsed_s: float
    metadata: dict[str, object] = {}


class CategorySummary(BaseModel):
    """Aggregated timing stats for one category."""
    category: str
    count: int
    total_s: float
    mean_s: float
    min_s: float
    max_s: float


class TimingProfile:
    """Thread-safe timing collector. No-op when disabled."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._entries: list[TimingEntry] = []
        self._lock = threading.Lock()
        self.cache_hits: int = 0
        self.cache_misses: int = 0

    @contextmanager
    def measure(self, category: str, label: str, **metadata: object) -> Iterator[None]:
        """Context manager that records elapsed time. No-op when disabled."""
        if not self.enabled:
            yield
            return
        start = time.monotonic()
        yield
        elapsed = time.monotonic() - start
        entry = TimingEntry(
            category=category, label=label,
            elapsed_s=elapsed, metadata=metadata,
        )
        with self._lock:
            self._entries.append(entry)

    def record_cache_hit(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.cache_hits += 1

    def record_cache_miss(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.cache_misses += 1

    def merge(self, other: TimingProfile) -> None:
        """Merge entries from a child profile."""
        with self._lock:
            self._entries.extend(other.entries)
            self.cache_hits += other.cache_hits
            self.cache_misses += other.cache_misses

    @property
    def entries(self) -> list[TimingEntry]:
        with self._lock:
            return list(self._entries)

    def summary(self) -> dict[str, CategorySummary]:
        """Aggregate entries by category."""
        groups: dict[str, list[float]] = defaultdict(list)
        for e in self.entries:
            groups[e.category].append(e.elapsed_s)
        result = {}
        for cat, times in groups.items():
            result[cat] = CategorySummary(
                category=cat,
                count=len(times),
                total_s=sum(times),
                mean_s=statistics.mean(times),
                min_s=min(times),
                max_s=max(times),
            )
        return result

    @staticmethod
    def print_summary(profile: TimingProfile, wall_s: float, console: Console) -> None:
        """Render timing summary as a Rich table."""
        summaries = profile.summary()
        if not summaries:
            return

        table = Table(title=f"Timing Breakdown ({wall_s:.1f}s wall clock)")
        table.add_column("Category")
        table.add_column("Count", justify="right")
        table.add_column("Total", justify="right")
        table.add_column("Mean", justify="right")
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")
        table.add_column("Wall %", justify="right")

        # Sort by total time descending
        for s in sorted(summaries.values(), key=lambda x: x.total_s, reverse=True):
            pct = (s.total_s / wall_s * 100) if wall_s > 0 else 0
            table.add_row(
                s.category, str(s.count),
                f"{s.total_s:.3f}s", f"{s.mean_s:.3f}s",
                f"{s.min_s:.3f}s", f"{s.max_s:.3f}s",
                f"{pct:.1f}%",
            )

        console.print(table)

        total_lookups = profile.cache_hits + profile.cache_misses
        if total_lookups > 0:
            hit_pct = profile.cache_hits / total_lookups * 100
            console.print(
                f"[dim]Cache: {profile.cache_hits} hits / "
                f"{total_lookups} lookups ({hit_pct:.1f}% hit rate)[/dim]"
            )
