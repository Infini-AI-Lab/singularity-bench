"""The harness.

Implement `run()`. Everything about how an agent works is yours to decide: the system prompt, what tools
exist and what they are called, how a tool result comes back, when to stop, what to do when something
fails. The one thing you do not build is the connection to the model — that is handed to you.

See /app/spec/HARNESS_SPEC.md.
"""
from __future__ import annotations

from typing import Any


def run(task: str, workspace: str, model: Any, budget: Any) -> None:
    """Drive an agent until the task is done or the budget is gone.

    task       what the user asked for, in plain language
    workspace  an absolute path; the project lives here
    model      a ModelClient (see the spec) -- the only way to reach a model
    budget     .calls_left(), .tokens_left(), .exhausted()

    Returns nothing. What matters is what happened: the workspace afterwards, and what you asked the
    model along the way. Must not raise.
    """
    raise NotImplementedError("implement run() — see /app/spec/HARNESS_SPEC.md")
