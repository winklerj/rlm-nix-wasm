"""Model pricing data and cost estimation."""

from __future__ import annotations

# Pricing per million tokens (as of Feb 2026)
MODEL_PRICING: dict[str, dict[str, float]] = {
    # Anthropic Claude models (https://platform.claude.com/docs/en/about-claude/pricing)
    "claude-opus-4-6": {"input": 5.00, "output": 25.00},
    "claude-opus-4-5": {"input": 5.00, "output": 25.00},
    "claude-opus-4-1": {"input": 15.00, "output": 75.00},
    "claude-opus-4": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},  # API ID alias
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},  # API ID alias
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},  # Claude Haiku 3.5
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},  # Claude Sonnet 3.5
    # OpenAI GPT text models, Standard tier (https://platform.openai.com/docs/pricing)
    "gpt-5.2": {"input": 1.75, "output": 14.00},
    "gpt-5.1": {"input": 1.25, "output": 10.00},
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in dollars based on model and token counts."""
    pricing = MODEL_PRICING.get(model, {"input": 1.00, "output": 5.00})
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost
