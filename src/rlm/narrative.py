"""Post-run narrative summary generation using Claude Opus 4.6."""

from __future__ import annotations

from litellm import completion


NARRATIVE_SYSTEM_PROMPT = """\
You are writing a diary-style narrative of an RLM (Recursive Language Model) \
investigation. Write in first person as the RLM system, describing the \
investigation step by step.

Structure your narrative as markdown with these sections:

## Goal
State the original query and the scale of the context that was investigated.

## Investigation
Write one paragraph per major phase of the investigation. For each phase describe:
- What operation was performed and why
- What was found or learned
- Any difficulties or dead ends encountered
- How findings informed the next step

Be specific about what the RLM actually did (sliced, grepped, chunked, etc.) \
and what patterns or information it found. Use concrete details from the steps.

## Conclusion
How the final answer was reached (or, if the run was stopped early, summarize \
what was accomplished and what the likely next steps would have been).

Keep the narrative concise but informative — aim for 300-600 words. \
Write in clear prose, not bullet points. The audience is a technical user \
who wants to understand the RLM's reasoning strategy.\
"""


def generate_narrative(
    *,
    query: str,
    context_len: int,
    model: str,
    steps: list[dict],
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

    response = completion(
        model="anthropic/claude-opus-4-6",
        messages=[
            {"role": "system", "content": NARRATIVE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.5,
        max_tokens=2048,
    )

    return response.choices[0].message.content or ""
