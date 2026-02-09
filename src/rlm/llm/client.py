"""LLM client abstraction using litellm."""

from __future__ import annotations

from litellm import completion

from rlm.types import RLMConfig


class LLMClient:
    """Manages LLM conversations for the explore/commit protocol."""

    def __init__(self, config: RLMConfig):
        self.config = config
        self.messages: list[dict[str, str]] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt for this conversation."""
        self.messages = [{"role": "system", "content": prompt}]

    def send(self, user_message: str) -> str:
        """Send a message and get the assistant's response."""
        self.messages.append({"role": "user", "content": user_message})

        response = completion(
            model=self.config.model,
            messages=self.messages,
            temperature=self.config.temperature,
        )

        # Track token usage
        if hasattr(response, 'usage') and response.usage:
            self.total_input_tokens += response.usage.prompt_tokens or 0
            self.total_output_tokens += response.usage.completion_tokens or 0

        assistant_message: str = response.choices[0].message.content or ""
        self.messages.append({"role": "assistant", "content": assistant_message})
        return assistant_message

    def message_count(self) -> int:
        """Number of messages in the conversation."""
        return len(self.messages)

    def get_token_usage(self) -> tuple[int, int]:
        """Return (input_tokens, output_tokens) for this conversation."""
        return self.total_input_tokens, self.total_output_tokens
