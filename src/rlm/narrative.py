"""Post-run narrative summary generation."""

from __future__ import annotations

from typing import Any, cast

from litellm import ModelResponse, completion
from litellm.types.utils import Choices

NARRATIVE_SYSTEM_PROMPT = """\
You are writing a structured implementation diary of an RLM (Recursive Language \
Model) investigation. Write in first person as the RLM system.

**Prose first**: every step starts with 1-2 short prose paragraphs before \
subsections. This is the "human story" that readers skim.

**Be specific**: include operation names, argument details, character counts, \
and concrete findings. Never be vague.

Use this exact markdown structure:

```
## Goal

Brief statement of the query and context scale.

## Step N: [Descriptive Name]

1-2 prose paragraphs: what this phase accomplished and what it unlocked.

### What I did
- Concrete operations run (op name, args, bind targets)

### Why
- Motivation for this approach

### What worked
- Key findings that moved the investigation forward

### What didn't work
- Dead ends, empty results, or surprising findings (with specifics)

### What I learned
- Insights, patterns discovered, corrected assumptions

### What was tricky
- Sharp edges: large context, ambiguous results, multiple candidates, etc.

## Conclusion

How the final answer was reached. If stopped early, what was accomplished and \
what the likely next steps would have been.
```

Group related operations into logical steps (e.g. "initial reconnaissance", \
"narrowing the search", "deep extraction"). Not every operation needs its own step — \
combine small sequential ops into one step when they serve the same purpose.

Omit empty subsections. If nothing didn't work, skip "What didn't work". \
Keep the diary scannable — short bullets, not long paragraphs in subsections.\
"""


def generate_narrative(
    *,
    query: str,
    context_len: int,
    model: str,
    steps: list[dict[str, Any]],
    final_answer: str | None,
    total_tokens: int,
    elapsed_s: float,
    completed: bool,
) -> str:
    """Generate a diary-style narrative of an RLM run.

    Args:
        query: The original user query.
        context_len: Length of the context in characters.
        model: The model used for the RLM run.
        steps: List of step dicts with keys: mode, summary, action, result, elapsed.
        final_answer: The final answer text, or None if stopped early.
        total_tokens: Total tokens consumed during the run.
        elapsed_s: Total wall-clock time in seconds.
        completed: Whether the run completed normally or was stopped.

    Returns:
        Markdown narrative text.
    """
    # Build the user prompt with run details
    status = "completed normally" if completed else "stopped early by the user"
    parts = [
        f"The RLM run {status}.",
        f"Query: {query}",
        f"Context size: {context_len:,} characters",
        f"Model: {model}",
        f"Total tokens: {total_tokens:,}",
        f"Elapsed time: {elapsed_s:.1f}s",
        "",
        "## Steps",
    ]

    for i, step in enumerate(steps, 1):
        mode = step.get("mode", "unknown").upper()
        summary = step.get("summary", "")
        result = step.get("result", "")
        step_elapsed = step.get("elapsed", "")

        # Truncate result for the prompt
        if isinstance(result, str) and len(result) > 500:
            result = result[:500] + "..."

        time_str = f" ({step_elapsed:.1f}s)" if isinstance(step_elapsed, (int, float)) else ""
        parts.append(f"Step {i} [{mode}]{time_str}: {summary}")
        if result:
            parts.append(f"  Result: {result}")

    parts.append("")
    if final_answer:
        parts.append(f"## Final Answer\n{final_answer}")
    else:
        parts.append("(No final answer — run was stopped before completion.)")

    user_prompt = "\n".join(parts)

    response = cast(
        ModelResponse,
        completion(
            model="anthropic/claude-sonnet-4",
            messages=[
                {"role": "system", "content": NARRATIVE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.5,
            max_tokens=2048,
        ),
    )

    choice = cast(Choices, response.choices[0])

    # Detect refusal or other non-stop finish reasons
    finish = getattr(choice, "finish_reason", None)
    if finish and finish not in ("stop", "end_turn", "length"):
        raise RuntimeError(f"Narrative generation refused by the model (finish_reason={finish!r})")

    content = choice.message.content or ""
    if not content.strip():
        raise RuntimeError("Narrative generation returned empty content")

    return content
