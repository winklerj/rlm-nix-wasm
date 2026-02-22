"""Tests for OOLONG benchmark scoring functions."""

from rlm.eval.scoring import parse_oolong_answer, score_oolong_synth


class TestParseOolongAnswer:
    def test_plain_string(self):
        assert parse_oolong_answer("entity") == "entity"

    def test_list_single(self):
        assert parse_oolong_answer("['entity']") == "entity"

    def test_list_double_quotes(self):
        assert parse_oolong_answer('["entity"]') == "entity"

    def test_list_multiple_takes_first(self):
        assert parse_oolong_answer("['a', 'b']") == "a"

    def test_whitespace_stripped(self):
        assert parse_oolong_answer("  hello  ") == "hello"

    def test_empty_string(self):
        assert parse_oolong_answer("") == ""

    def test_numeric_string(self):
        assert parse_oolong_answer("42") == "42"

    def test_list_with_number(self):
        assert parse_oolong_answer("[42]") == "42"

    def test_malformed_list(self):
        assert parse_oolong_answer("['unclosed") == "['unclosed"

    def test_plain_with_brackets(self):
        # Not valid Python literal, returns as-is
        assert parse_oolong_answer("[not a list") == "[not a list"


class TestScoreExactMatch:
    def test_exact_match_any_type(self):
        assert score_oolong_synth("foo", "foo", "label") == 1.0
        assert score_oolong_synth("42", "42", "numeric") == 1.0
        assert score_oolong_synth("user123", "user123", "user") == 1.0

    def test_empty_predicted(self):
        assert score_oolong_synth("", "foo", "label") == 0.0

    def test_whitespace_handling(self):
        assert score_oolong_synth("  foo  ", "foo", "label") == 1.0


class TestScoreNumeric:
    def test_exact(self):
        assert score_oolong_synth("10", "10", "numeric") == 1.0

    def test_off_by_one(self):
        assert score_oolong_synth("11", "10", "numeric") == 0.75

    def test_off_by_two(self):
        assert score_oolong_synth("12", "10", "numeric") == 0.75 ** 2

    def test_off_by_five(self):
        score = score_oolong_synth("15", "10", "numeric")
        assert abs(score - 0.75 ** 5) < 1e-9

    def test_non_numeric_predicted(self):
        assert score_oolong_synth("abc", "10", "numeric") == 0.0

    def test_non_numeric_gold(self):
        assert score_oolong_synth("10", "abc", "numeric") == 0.0

    def test_both_non_numeric(self):
        assert score_oolong_synth("abc", "xyz", "numeric") == 0.0

    def test_negative_numbers(self):
        assert score_oolong_synth("-5", "-5", "numeric") == 1.0

    def test_numeric_with_commas(self):
        # Strips non-digit chars (except -)
        assert score_oolong_synth("1,000", "1000", "numeric") == 1.0


class TestScoreLabel:
    def test_case_insensitive(self):
        assert score_oolong_synth("Entity", "entity", "label") == 1.0

    def test_mismatch(self):
        assert score_oolong_synth("person", "entity", "label") == 0.0

    def test_case_insensitive_both_upper(self):
        assert score_oolong_synth("ENTITY", "entity", "label") == 1.0


class TestScoreComparison:
    def test_exact(self):
        assert score_oolong_synth("more common", "more common", "comparison") == 1.0

    def test_synonym(self):
        assert score_oolong_synth("more frequent", "more common", "comparison") == 1.0

    def test_same_frequency_synonyms(self):
        assert score_oolong_synth("equal frequency", "same frequency", "comparison") == 1.0

    def test_mismatch(self):
        assert score_oolong_synth("more common", "less common", "comparison") == 0.0

    def test_case_insensitive(self):
        assert score_oolong_synth("More Common", "more common", "comparison") == 1.0

    def test_gold_with_than_suffix(self):
        """OOLONG gold answers use 'more common than' format."""
        assert score_oolong_synth("more common", "more common than", "comparison") == 1.0
        assert score_oolong_synth("less common", "less common than", "comparison") == 1.0

    def test_gold_with_as_suffix(self):
        assert score_oolong_synth("same frequency", "same frequency as", "comparison") == 1.0

    def test_verbose_prediction_with_than_gold(self):
        """Model outputs like 'Answer: X is more common than Y' with gold='more common than'."""
        pred = "Answer: description and abstract concept is more common than numeric value"
        assert score_oolong_synth(pred, "more common than", "comparison") == 1.0

    def test_verbose_wrong_with_than_gold(self):
        pred = "Answer: description and abstract concept is more common than entity"
        assert score_oolong_synth(pred, "less common than", "comparison") == 0.0


class TestScoreDate:
    def test_exact_string(self):
        assert score_oolong_synth("2024-01-15", "2024-01-15", "date") == 1.0

    def test_different_format(self):
        assert score_oolong_synth("January 15, 2024", "2024-01-15", "date") == 1.0

    def test_mismatch(self):
        assert score_oolong_synth("2024-01-16", "2024-01-15", "date") == 0.0

    def test_month_year_type(self):
        assert score_oolong_synth("January 2024", "January 2024", "month_year") == 1.0

    def test_invalid_date(self):
        assert score_oolong_synth("not a date", "2024-01-15", "date") == 0.0


class TestScoreUser:
    def test_exact(self):
        assert score_oolong_synth("user123", "user123", "user") == 1.0

    def test_mismatch(self):
        assert score_oolong_synth("user123", "user456", "user") == 0.0

    def test_case_sensitive(self):
        # User IDs are case-sensitive
        assert score_oolong_synth("User123", "user123", "user") == 0.0

    def test_user_prefix_stripped(self):
        assert score_oolong_synth("User: user123", "user123", "user") == 1.0

    def test_user_prefix_lowercase(self):
        assert score_oolong_synth("user: user123", "user123", "user") == 1.0

    def test_answer_prefix_stripped(self):
        assert score_oolong_synth("Answer: user123", "user123", "user") == 1.0

    def test_answer_prefix_lowercase(self):
        assert score_oolong_synth("answer: user123", "user123", "user") == 1.0

    def test_prefix_with_wrong_user(self):
        assert score_oolong_synth("User: user456", "user123", "user") == 0.0

    def test_prefix_with_whitespace(self):
        assert score_oolong_synth("  User:  user123  ", "user123", "user") == 1.0


class TestScoreUnknownType:
    def test_case_insensitive_match(self):
        assert score_oolong_synth("Foo", "foo", "unknown_type") == 1.0

    def test_mismatch(self):
        assert score_oolong_synth("bar", "foo", "unknown_type") == 0.0
