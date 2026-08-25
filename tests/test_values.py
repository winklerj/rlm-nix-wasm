"""Tests for tolerant list-value parsing of op outputs."""

from __future__ import annotations

import json

from rlm.ops.values import parse_list_value


def test_json_list() -> None:
    assert parse_list_value(json.dumps(["a", "b"])) == ["a", "b"]


def test_python_repr_list_from_eval() -> None:
    # eval prints Python reprs with single quotes, which is not JSON.
    assert parse_list_value("['What is X?', 'Who is Y?']") == ["What is X?", "Who is Y?"]


def test_non_string_items_are_stringified() -> None:
    assert parse_list_value("[1, 2, 3]") == ["1", "2", "3"]


def test_surrounding_whitespace() -> None:
    assert parse_list_value("  ['a']\n") == ["a"]


def test_plain_text_is_single_item() -> None:
    assert parse_list_value("just some text") == ["just some text"]


def test_unparseable_bracket_text_is_single_item() -> None:
    raw = "[not really a list"
    assert parse_list_value(raw) == [raw]
