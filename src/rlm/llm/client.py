"""LLM client abstraction using litellm."""

from __future__ import annotations

import json
import time
from typing import Any

from litellm import completion
from rich.console import Console

from rlm.timing import TimingProfile
from rlm.trace import OrchestratorTrace, TraceCollector
from rlm.types import RLMConfig


def _is_tool_format_error(exc: Exception) -> bool:
    """True if the server rejected the model's tool-call syntax (HTTP 500)."""
    text = str(exc).lower()
    return (
        "peg-native" in text
        or "does not match the expected" in text
        or "failed to parse" in text and "tool" in text
    )


class LLMRefusalError(Exception):
    """The LLM refused to respond (e.g. safety filter triggered)."""
    pass


class LLMClient:
    """Manages LLM conversations for the explore/commit protocol."""

    def __init__(self, config: RLMConfig, profile: TimingProfile | None = None,
                 verbose: bool = False, console: Console | None = None,
                 trace: TraceCollector | None = None,
                 trace_node: OrchestratorTrace | None = None,
                 use_tools: bool = True):
        self.config = config
        # When False, the model gets no tool definitions and its plain text
        # reply is returned as-is (direct prompting, e.g. baseline / leaf calls).
        self.use_tools = use_tools
        self.messages: list[dict[str, str]] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.profile = profile or TimingProfile()
        self.verbose = verbose
        self.console = console or Console(stderr=True)
        self._call_count = 0
        self._trace = trace or TraceCollector()
        self._trace_node = trace_node

    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "rlm_explore",
                "description": "Perform a single explore operation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "op": {"type": "string", "enum": ["slice","grep","count","chunk","split","eval"]},
                        "args": {"type": "object"},
                        "bind": {"type": "string"}
                    },
                    "required": ["op","args","bind"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "rlm_commit",
                "description": (
                    "Execute a multi-operation plan as a batch. Required for "
                    "chunk/map/combine workflows (e.g. counting or classifying "
                    "across the whole context)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "op": {"type": "string", "enum": ["slice","grep","count","chunk","split","eval","combine","rlm_call","map","calibrated_tally"]},
                                    "args": {"type": "object"},
                                    "bind": {"type": "string"}
                                },
                                "required": ["op","args","bind"]
                            }
                        },
                        "output": {"type": "string", "description": "Name of the bound variable to return as the plan result."}
                    },
                    "required": ["operations","output"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "rlm_final",
                "description": "Return final answer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"}
                    },
                    "required": ["answer"]
                }
            }
        }
    ]


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

        # Prune conversation history to avoid context overflow
        if len(self.messages) > 32:
            system_msg = self.messages[0]
            self.messages = [system_msg] + self.messages[-30:]

        with self.profile.measure("llm", "send", model=self.config.model):
            start = time.monotonic()
            tool_kwargs: dict[str, Any] = (
                {"tools": self.TOOLS, "tool_choice": "auto"} if self.use_tools else {}
            )
            if self.config.max_output_tokens:
                tool_kwargs["max_tokens"] = self.config.max_output_tokens
            if self.config.reasoning_strength:
                tool_kwargs["extra_body"] = {
                    "chat_template_kwargs": {
                        "reasoning_strength": self.config.reasoning_strength
                    }
                }
            # Some servers (llama-server) reject a malformed tool call from the
            # model with a 500 rather than returning it. At temperature 0 an
            # identical retry reproduces it, so escalate: (1) drop the tool
            # definitions so the model answers in plain JSON (which the parser
            # understands); (2) if it still emits broken native tool syntax
            # (steered by earlier tool calls in the history), add a corrective
            # user turn that changes the prompt and states the format.
            response = None
            for attempt in range(3):
                try:
                    response = completion(
                        model=self.config.model,
                        messages=self.messages,
                        temperature=self.config.temperature,
                        **tool_kwargs,
                    )
                    break
                except Exception as e:
                    if not (self.use_tools and _is_tool_format_error(e)) or attempt == 2:
                        raise
                    if self.verbose:
                        self.console.print(
                            "[yellow]  tool-call format rejected by server; "
                            f"retry {attempt + 1}/2 "
                            f"({'without tools' if attempt == 0 else 'with format nudge'})"
                            "[/yellow]"
                        )
                    tool_kwargs.pop("tools", None)
                    tool_kwargs.pop("tool_choice", None)
                    if attempt == 1:
                        self.messages.append({
                            "role": "user",
                            "content": (
                                "Your previous response was malformed and could not be "
                                "parsed. Do NOT use tool-call syntax. Respond with a "
                                "single raw JSON object with a 'mode' field "
                                "(explore, commit, or final)."
                            ),
                        })
            assert response is not None
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

        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason and finish_reason not in ("stop", "end_turn", "length", "tool_calls"):
            raise LLMRefusalError(
                f"LLM refused to respond (finish_reason={finish_reason!r})"
            )

        message = choice.message
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            tc = tool_calls[0]
            fn_name = tc.function.name
            args = json.loads(tc.function.arguments)
            # Models don't always honour "required" in the schema, so read
            # fields leniently; the parser reports what's actually missing.
            if fn_name == "rlm_explore":
                action_json = json.dumps({
                    "mode": "explore",
                    "operation": {
                        "op": args.get("op"),
                        "args": args.get("args", {}),
                        "bind": args.get("bind"),
                    }
                })
            elif fn_name == "rlm_commit":
                action_json = json.dumps({
                    "mode": "commit",
                    "operations": args.get("operations", []),
                    "output": args.get("output"),
                })
            elif fn_name == "rlm_final":
                action_json = json.dumps({
                    "mode": "final",
                    "answer": args.get("answer", ""),
                })
            else:
                action_json = "{}"
            assistant_message = action_json
            self.messages.append({"role": "assistant", "content": action_json, "tool_calls": tool_calls})
        else:
            assistant_message = message.content or ""
            if not assistant_message.strip():
                if finish_reason == "length":
                    raise LLMRefusalError(
                        "LLM hit the output token limit before producing an "
                        f"answer ({call_out:,} tokens, likely all reasoning); "
                        "raise max_output_tokens or lower reasoning_strength"
                    )
                raise LLMRefusalError("LLM returned empty content")
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
