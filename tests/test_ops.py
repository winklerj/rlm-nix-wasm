"""Tests for DSL text operations and combine."""

import json

from rlm.ops.text import op_chunk, op_count, op_grep, op_slice, op_split
from rlm.ops.recursive import op_combine


SAMPLE_TEXT = "line one\nline two\nline three\nline four\nline five"


def _b(text: str, name: str = "ctx") -> dict[str, str]:
    """Helper to create bindings."""
    return {name: text}


# --- slice ---

class TestSlice:
    def test_basic(self):
        result = op_slice({"input": "ctx", "start": 0, "end": 8}, _b(SAMPLE_TEXT))
        assert result == "line one"

    def test_defaults(self):
        result = op_slice({"input": "ctx"}, _b(SAMPLE_TEXT))
        assert result == SAMPLE_TEXT

    def test_middle(self):
        result = op_slice({"input": "ctx", "start": 9, "end": 17}, _b(SAMPLE_TEXT))
        assert result == "line two"

    def test_empty_input(self):
        result = op_slice({"input": "ctx", "start": 0, "end": 5}, _b(""))
        assert result == ""

    def test_out_of_bounds(self):
        result = op_slice({"input": "ctx", "start": 0, "end": 1000}, _b("short"))
        assert result == "short"


# --- grep ---

class TestGrep:
    def test_literal(self):
        result = op_grep({"input": "ctx", "pattern": "two"}, _b(SAMPLE_TEXT))
        assert result == "line two"

    def test_multiple_matches(self):
        result = op_grep({"input": "ctx", "pattern": "line"}, _b(SAMPLE_TEXT))
        assert result == SAMPLE_TEXT

    def test_no_match(self):
        result = op_grep({"input": "ctx", "pattern": "xyz"}, _b(SAMPLE_TEXT))
        assert result == ""

    def test_regex_pattern(self):
        result = op_grep({"input": "ctx", "pattern": r"t\w+e"}, _b(SAMPLE_TEXT))
        assert "three" in result

    def test_regex_digit(self):
        text = "abc\n123\ndef\n456"
        result = op_grep({"input": "ctx", "pattern": r"\d+"}, _b(text))
        assert result == "123\n456"


# --- count ---

class TestCount:
    def test_lines(self):
        result = op_count({"input": "ctx", "mode": "lines"}, _b(SAMPLE_TEXT))
        assert result == "5"

    def test_lines_default(self):
        result = op_count({"input": "ctx"}, _b(SAMPLE_TEXT))
        assert result == "5"

    def test_chars(self):
        result = op_count({"input": "ctx", "mode": "chars"}, _b(SAMPLE_TEXT))
        assert result == str(len(SAMPLE_TEXT))

    def test_empty(self):
        result = op_count({"input": "ctx"}, _b(""))
        assert result == "0"

    def test_single_line(self):
        result = op_count({"input": "ctx"}, _b("hello"))
        assert result == "1"


# --- split ---

class TestSplit:
    def test_default_delimiter(self):
        result = op_split({"input": "ctx"}, _b(SAMPLE_TEXT))
        parts = json.loads(result)
        assert len(parts) == 5
        assert parts[0] == "line one"

    def test_custom_delimiter(self):
        text = "a||b||c"
        result = op_split({"input": "ctx", "delimiter": "||"}, _b(text))
        parts = json.loads(result)
        assert parts == ["a", "b", "c"]

    def test_single_item(self):
        result = op_split({"input": "ctx", "delimiter": ","}, _b("no commas"))
        parts = json.loads(result)
        assert parts == ["no commas"]


# --- chunk ---

class TestChunk:
    def test_even_split(self):
        text = "a\nb\nc\nd"
        result = op_chunk({"input": "ctx", "n": 2}, _b(text))
        chunks = json.loads(result)
        assert len(chunks) == 2
        assert chunks[0] == "a\nb"
        assert chunks[1] == "c\nd"

    def test_uneven_split(self):
        result = op_chunk({"input": "ctx", "n": 3}, _b(SAMPLE_TEXT))
        chunks = json.loads(result)
        assert len(chunks) == 3
        # All lines should be present across chunks
        all_lines = "\n".join(chunks).split("\n")
        assert len(all_lines) == 5

    def test_more_chunks_than_lines(self):
        text = "a\nb"
        result = op_chunk({"input": "ctx", "n": 5}, _b(text))
        chunks = json.loads(result)
        # Should still produce chunks (some may be empty-ish)
        assert len(chunks) >= 2

    def test_single_chunk(self):
        result = op_chunk({"input": "ctx", "n": 1}, _b(SAMPLE_TEXT))
        chunks = json.loads(result)
        assert len(chunks) == 1
        assert chunks[0] == SAMPLE_TEXT


# --- combine ---

class TestCombine:
    def test_concat_with_list(self):
        bindings = {"a": "hello", "b": "world"}
        result = op_combine({"inputs": ["a", "b"], "strategy": "concat"}, bindings)
        assert result == "hello\nworld"

    def test_concat_with_json_binding(self):
        bindings = {"arr": json.dumps(["x", "y", "z"])}
        result = op_combine({"inputs": "arr", "strategy": "concat"}, bindings)
        assert result == "x\ny\nz"

    def test_sum(self):
        bindings = {"arr": json.dumps(["10", "20", "30"])}
        result = op_combine({"inputs": "arr", "strategy": "sum"}, bindings)
        assert result == "60"

    def test_sum_with_non_digits(self):
        bindings = {"arr": json.dumps(["10", "abc", "20"])}
        result = op_combine({"inputs": "arr", "strategy": "sum"}, bindings)
        assert result == "30"

    def test_vote(self):
        bindings = {"arr": json.dumps(["yes", "no", "yes", "yes"])}
        result = op_combine({"inputs": "arr", "strategy": "vote"}, bindings)
        assert result == "yes"

    def test_default_strategy(self):
        bindings = {"a": "x", "b": "y"}
        result = op_combine({"inputs": ["a", "b"]}, bindings)
        assert result == "x\ny"
