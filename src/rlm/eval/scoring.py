"""OOLONG benchmark scoring functions.

Implements the scoring methodology from the OOLONG paper.
"""

from __future__ import annotations

import ast
import re


def parse_oolong_answer(raw: str) -> str:
    """Unwrap OOLONG answer format.

    Answers may be wrapped in list format like "['entity']" or plain strings.
    Uses ast.literal_eval which is safe — it only parses Python literals
    (strings, numbers, lists, tuples, dicts, booleans, None), not arbitrary code.

    Args:
        raw: Raw answer string from the dataset or model output.

    Returns:
        Unwrapped answer string, stripped of whitespace.
    """
    raw = raw.strip()
    if not raw:
        return ""

    # Try to unwrap list format: "['entity']" -> "entity"
    try:
        parsed = ast.literal_eval(raw)  # noqa: S307 — safe: only parses literals
        if isinstance(parsed, list) and len(parsed) >= 1:
            return str(parsed[0]).strip()
    except (ValueError, SyntaxError):
        pass

    return raw


# Comparison synonyms for OOLONG comparison-type questions
_COMPARISON_SYNONYMS: dict[str, set[str]] = {
    "more common": {
        "more common",
        "more common than",
        "more frequent",
        "higher frequency",
    },
    "less common": {
        "less common",
        "less common than",
        "less frequent",
        "lower frequency",
    },
    "same frequency": {
        "same frequency",
        "same frequency as",
        "equal frequency",
        "same",
    },
}


def _normalize_comparison(value: str) -> str:
    """Normalize comparison answer to canonical form."""
    value_lower = value.lower().strip()
    for canonical, synonyms in _COMPARISON_SYNONYMS.items():
        if value_lower in synonyms:
            return canonical
    return value_lower


def score_oolong_synth(predicted: str, gold_answer: str, answer_type: str) -> float:
    """Score a predicted answer against the gold answer using OOLONG methodology.

    Args:
        predicted: The model's predicted answer (already parsed/unwrapped).
        gold_answer: The gold standard answer (already parsed/unwrapped).
        answer_type: The OOLONG answer type (e.g., "numeric", "label", "comparison",
            "date", "month_year", "user").

    Returns:
        Score between 0.0 and 1.0.
    """
    predicted = predicted.strip()
    gold_answer = gold_answer.strip()

    if not predicted:
        return 0.0

    # Exact match always scores 1.0
    if predicted == gold_answer:
        return 1.0

    answer_type_lower = answer_type.lower()

    if answer_type_lower == "numeric":
        return _score_numeric(predicted, gold_answer)
    elif answer_type_lower == "label":
        return _score_label(predicted, gold_answer)
    elif answer_type_lower == "comparison":
        return _score_comparison(predicted, gold_answer)
    elif answer_type_lower in ("date", "month_year"):
        return _score_date(predicted, gold_answer)
    elif answer_type_lower == "user":
        return _score_user(predicted, gold_answer)
    else:
        # Unknown type: case-insensitive exact match
        return 1.0 if predicted.lower() == gold_answer.lower() else 0.0


def _score_numeric(predicted: str, gold: str) -> float:
    """Score numeric answers with exponential decay: 0.75 ** |gold - predicted|."""
    try:
        pred_int = int(re.sub(r"[^\d\-]", "", predicted))
        gold_int = int(re.sub(r"[^\d\-]", "", gold))
    except (ValueError, TypeError):
        return 0.0
    return 0.75 ** abs(gold_int - pred_int)


def _score_label(predicted: str, gold: str) -> float:
    """Score label answers with case-insensitive exact match.

    Handles common model output prefixes like "Label: entity" or "Answer: entity".
    """
    pred = predicted.lower().strip()
    gold_lower = gold.lower().strip()

    # Direct match
    if pred == gold_lower:
        return 1.0

    # Strip common prefixes: "Label: X", "Answer: X"
    for prefix in ("label:", "answer:", "the answer is", "final answer:"):
        if pred.startswith(prefix):
            pred = pred[len(prefix):].strip()
            if pred == gold_lower:
                return 1.0

    # Check if gold appears as a substring (model may elaborate)
    if gold_lower in pred:
        return 1.0

    return 0.0


def _score_comparison(predicted: str, gold: str) -> float:
    """Score comparison answers with synonym matching.

    Handles verbose model outputs by searching for comparison keywords
    within the predicted text.
    """
    norm_pred = _normalize_comparison(predicted)
    norm_gold = _normalize_comparison(gold)
    if norm_pred == norm_gold:
        return 1.0

    # If the predicted text is verbose, search for comparison keywords
    pred_lower = predicted.lower().strip()
    for canonical, synonyms in _COMPARISON_SYNONYMS.items():
        for syn in synonyms:
            if syn in pred_lower:
                if canonical == norm_gold:
                    return 1.0
                else:
                    return 0.0  # Found a comparison keyword but it's wrong

    # Also match "more common than" / "less common than" patterns in gold
    if norm_gold in pred_lower:
        return 1.0

    return 0.0


def _score_date(predicted: str, gold: str) -> float:
    """Score date/month_year answers with date parsing + exact match."""
    try:
        from dateutil.parser import parse as date_parse
        pred_date = date_parse(predicted, fuzzy=True)
        gold_date = date_parse(gold, fuzzy=True)
        return 1.0 if pred_date == gold_date else 0.0
    except (ValueError, ImportError):
        # Fallback: case-insensitive string match
        return 1.0 if predicted.lower().strip() == gold.lower().strip() else 0.0


def _score_user(predicted: str, gold: str) -> float:
    """Score user ID answers with exact match.

    Handles common model output prefixes like "User: user123" or "Answer: user123".
    """
    pred = predicted.strip()
    gold_stripped = gold.strip()

    if pred == gold_stripped:
        return 1.0

    # Strip common prefixes: "User: X", "Answer: X", "user: X"
    for prefix in ("user:", "answer:", "User:", "Answer:"):
        if pred.startswith(prefix):
            pred = pred[len(prefix):].strip()
            if pred == gold_stripped:
                return 1.0

    return 0.0
