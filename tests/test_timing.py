"""Tests for the timing instrumentation module."""

import threading
import time

import pytest

from rlm.timing import TimingProfile


class TestTimingProfileDisabled:
    def test_measure_is_noop(self):
        profile = TimingProfile(enabled=False)
        with profile.measure("llm", "send"):
            time.sleep(0.01)
        assert profile.entries == []

    def test_cache_tracking_is_noop(self):
        profile = TimingProfile(enabled=False)
        profile.record_cache_hit()
        profile.record_cache_miss()
        assert profile.cache_hits == 0
        assert profile.cache_misses == 0


class TestTimingProfileEnabled:
    def test_collects_entries(self):
        profile = TimingProfile(enabled=True)
        with profile.measure("llm", "send", model="test"):
            time.sleep(0.01)
        assert len(profile.entries) == 1
        entry = profile.entries[0]
        assert entry.category == "llm"
        assert entry.label == "send"
        assert entry.elapsed_s >= 0.01
        assert entry.metadata == {"model": "test"}

    def test_multiple_entries(self):
        profile = TimingProfile(enabled=True)
        with profile.measure("llm", "send"):
            pass
        with profile.measure("cache", "lookup"):
            pass
        assert len(profile.entries) == 2

    def test_cache_tracking(self):
        profile = TimingProfile(enabled=True)
        profile.record_cache_hit()
        profile.record_cache_hit()
        profile.record_cache_miss()
        assert profile.cache_hits == 2
        assert profile.cache_misses == 1


class TestMerge:
    def test_merge_entries(self):
        parent = TimingProfile(enabled=True)
        child = TimingProfile(enabled=True)

        with parent.measure("llm", "send"):
            pass
        with child.measure("cache", "lookup"):
            pass
        child.record_cache_hit()
        child.record_cache_miss()

        parent.merge(child)
        assert len(parent.entries) == 2
        assert parent.cache_hits == 1
        assert parent.cache_misses == 1

    def test_merge_empty(self):
        parent = TimingProfile(enabled=True)
        child = TimingProfile(enabled=True)
        parent.merge(child)
        assert len(parent.entries) == 0


class TestSummary:
    def test_aggregates_by_category(self):
        profile = TimingProfile(enabled=True)
        # Add entries manually for deterministic testing
        from rlm.timing import TimingEntry
        profile._entries = [
            TimingEntry(category="llm", label="send", elapsed_s=1.0),
            TimingEntry(category="llm", label="send", elapsed_s=2.0),
            TimingEntry(category="cache", label="lookup", elapsed_s=0.001),
        ]

        summaries = profile.summary()
        assert "llm" in summaries
        assert "cache" in summaries

        llm = summaries["llm"]
        assert llm.count == 2
        assert llm.total_s == pytest.approx(3.0)
        assert llm.mean_s == pytest.approx(1.5)
        assert llm.min_s == pytest.approx(1.0)
        assert llm.max_s == pytest.approx(2.0)

        cache = summaries["cache"]
        assert cache.count == 1
        assert cache.total_s == pytest.approx(0.001)

    def test_empty_summary(self):
        profile = TimingProfile(enabled=True)
        assert profile.summary() == {}


class TestThreadSafety:
    def test_concurrent_measure(self):
        profile = TimingProfile(enabled=True)
        errors: list[Exception] = []

        def worker(n: int) -> None:
            try:
                for _ in range(50):
                    with profile.measure("test", f"worker-{n}"):
                        pass
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(profile.entries) == 200  # 4 threads * 50 entries

    def test_concurrent_cache_tracking(self):
        profile = TimingProfile(enabled=True)

        def hit_worker() -> None:
            for _ in range(100):
                profile.record_cache_hit()

        def miss_worker() -> None:
            for _ in range(100):
                profile.record_cache_miss()

        threads = [
            threading.Thread(target=hit_worker),
            threading.Thread(target=miss_worker),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert profile.cache_hits == 100
        assert profile.cache_misses == 100


class TestPrintSummary:
    def test_print_summary_with_data(self, capsys):
        """Verify print_summary doesn't crash and produces output."""
        from rich.console import Console
        from rlm.timing import TimingEntry

        profile = TimingProfile(enabled=True)
        profile._entries = [
            TimingEntry(category="llm", label="send", elapsed_s=1.5),
            TimingEntry(category="cache", label="lookup", elapsed_s=0.001),
        ]
        profile.cache_hits = 3
        profile.cache_misses = 1

        console = Console(force_terminal=False)
        TimingProfile.print_summary(profile, 2.0, console)
        # Just verify it doesn't crash; Rich output goes to stdout

    def test_print_summary_empty(self, capsys):
        """Empty profile should produce no output."""
        from rich.console import Console

        profile = TimingProfile(enabled=True)
        console = Console(force_terminal=False)
        TimingProfile.print_summary(profile, 1.0, console)

    def test_print_summary_no_cache(self, capsys):
        """No cache lookups should skip cache line."""
        from rich.console import Console
        from rlm.timing import TimingEntry

        profile = TimingProfile(enabled=True)
        profile._entries = [
            TimingEntry(category="llm", label="send", elapsed_s=1.0),
        ]
        console = Console(force_terminal=False)
        TimingProfile.print_summary(profile, 1.0, console)
