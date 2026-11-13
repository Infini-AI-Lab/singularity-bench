"""The two objects `run()` is handed. Shipped so the shapes are unambiguous; do not modify."""
from __future__ import annotations

from typing import Any


class ModelClient:
    """The model your harness is driving: the one that decides what to do next.

    This is not the only model available. `HARNESS_LLM_BASE_URL` (see the spec) is a separate served
    endpoint your harness may call for its own internal work -- deciding what a summary should keep,
    judging whether a result is worth showing. That one is optional; this one is the agent itself.

    complete(messages, tools=None, **kw) -> an OpenAI-shaped response dict:

        {"choices": [{"message": {"role": "assistant",
                                  "content": str | None,
                                  "tool_calls": [{"id": str, "type": "function",
                                                  "function": {"name": str, "arguments": str}}]}}]}

    `messages` is a list of {"role", "content"} dicts, with the usual roles. `tools` is a list of
    OpenAI-shaped tool schemas. Describe your tools properly: what the model is offered is also how the
    grader knows what your harness can do.
    """

    def complete(self, messages: list[dict], tools: list[dict] | None = None, **kw: Any) -> dict:
        raise NotImplementedError


class Budget:
    """What is left. Ask before spending; stop when it is gone.

    **The budget is time.** `seconds_left()` is the one that runs out, and `exhausted()` becomes true
    when it does. Requirement 6 -- leave the workspace in a state someone could pick up -- is only
    satisfiable if you watch it.

    `calls_left()` and `tokens_left()` are here so a harness written against a deployment that meters
    those does not crash. **This deployment does not limit either**, and they return a large sentinel
    to say so. A harness that waits for `calls_left()` to fall will wait forever; watch the clock.
    """

    def seconds_left(self) -> float: raise NotImplementedError
    def exhausted(self) -> bool: raise NotImplementedError

    # Compatibility only -- unbounded here. See the class docstring.
    def calls_left(self) -> int: raise NotImplementedError
    def tokens_left(self) -> int: raise NotImplementedError
