"""One multi-turn terminal trajectory.

response_ids includes the interleaved tool observations. action_mask selects only
the tokens the policy actually sampled, not a contiguous response span.
"""
from dataclasses import dataclass, field
import math

@dataclass
class Sample:
    prompt: str
    prompt_ids: list[int]
    response_ids: list[int]
    rollout_logprobs: list[float]
    action_mask: list[int]
    reward: float
    group_id: str
    policy_version: str
    metadata: dict = field(default_factory=dict)
    advantage: float | None = None

    def validate(self):
        if not self.prompt_ids or not self.response_ids:
            raise ValueError('Nonempty prompt and response required')
        n = len(self.response_ids)
        if len(self.rollout_logprobs) != n or len(self.action_mask) != n:
            raise ValueError('Misaligned response IDs, logprobs or action mask')
        if any(type(t) is not int or t < 0 for t in self.prompt_ids+self.response_ids):
            raise ValueError('Invalid token IDs')
        if any(type(m) is not int or m not in (0,1) for m in self.action_mask) or not any(self.action_mask):
            raise ValueError('Invalid/empty action mask')
        if not all(math.isfinite(v) for v in self.rollout_logprobs) or not math.isfinite(self.reward):
            raise ValueError('Nonfinite reward/logprobs')
        if any(p > 1e-6 for p,m in zip(self.rollout_logprobs,self.action_mask) if m):
            raise ValueError('Action logprob cannot be positive')
        if self.reward not in (0.0,1.0):
            raise ValueError('This dataset expects a binary verifier reward')
        if not self.group_id or not self.policy_version:
            raise ValueError('Group ID and observed policy version required')
        if self.advantage is not None and not math.isfinite(self.advantage):
            raise ValueError('Nonfinite advantage')
