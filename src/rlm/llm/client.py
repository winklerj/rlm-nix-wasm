"""LLM client abstraction using litellm."""

from __future__ import annotations

import time

from litellm import completion
from rich.console import Console

from rlm.timing import TimingProfile
from rlm.trace import OrchestratorTrace, TraceCollector
from rlm.types import RLMConfig


class LLMClient:
    """Manages LLM conversations for the explore/commit protocol."""

    def __init__(self, config: RLMConfig, profile: TimingProfile | None = None,
                 verbose: bool = False, console: Console | None = None,
                 trace: TraceCollector | None = None,
                 trace_node: OrchestratorTrace | None = None):
        self.config = config
        self.messages: list[dict[str, str]] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.profile = profile or TimingProfile()
        self.verbose = verbose
        self.console = console or Console(stderr=True)
        self._call_count = 0
        self._trace = trace or TraceCollector()
        self._trace_node = trace_node

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt for this conversation."""
        self.messages = [{"role": "system", "content": prompt}]

    def send(self, user_message: str) -> str:
        """Send a message and get the assistant's response."""
        self.messages.append({"role": "user", "content": user_message})
        self._call_count += 1
        call_num = self._call_count

        if self.verbose:
            self.console.print(
                f"[yellow]  LLM call #{call_num} ({self.config.model})…[/yellow]"
            )

        with self.profile.measure("llm", "send", model=self.config.model):
            start = time.monotonic()
            response = completion(
                model=self.config.model,
                messages=self.messages,
                temperature=self.config.temperature,
            )
            elapsed = time.monotonic() - start

        # Track token usage
        call_in = 0
        call_out = 0
        if hasattr(response, 'usage') and response.usage:
            call_in = response.usage.prompt_tokens or 0
            call_out = response.usage.completion_tokens or 0
            self.total_input_tokens += call_in
            self.total_output_tokens += call_out

        if self.verbose:
            self.console.print(
                f"[yellow]  LLM call #{call_num}: {elapsed:.1f}s, "
                f"{call_in:,} in + {call_out:,} out tokens[/yellow]"
            )

        assistant_message: str = response.choices[0].message.content or ""
        self.messages.append({"role": "assistant", "content": assistant_message})

        if self._trace_node is not None:
            self._trace.record_llm_call(
                self._trace_node,
                call_number=call_num,
                elapsed_s=elapsed,
                model=self.config.model,
                input_tokens=call_in,
                output_tokens=call_out,
                user_message=user_message,
                assistant_message=assistant_message,
            )

        return assistant_message

    def message_count(self) -> int:
        """Number of messages in the conversation."""
        return len(self.messages)

    def get_token_usage(self) -> tuple[int, int]:
        """Return (input_tokens, output_tokens) for this conversation."""
        return self.total_input_tokens, self.total_output_tokens
