"""Terminal profiles. GPU/model compatibility must be tested on the allocation."""
from dataclasses import dataclass
import re

MODEL_ID = 'zai-org/GLM-4.7-Flash'
MODEL_REVISION = '7dd20894a642a0aa287e9827cb1a1f7f91386b67'

@dataclass(frozen=True)
class Profile:
    gpu_type: str
    context: int
    response: int
    samples: int
    concurrency: int
    engine_tp: int
    episode_seconds: int

PROFILES = {
    'l40-debug': Profile('L40', 4096, 1024, 4, 4, 4, 1800),
    'h200-debug': Profile('H200', 8192, 2048, 4, 4, 4, 600),
    'gb300': Profile('GB300', 65536, 8192, 8, 32, 1, 5400),
}

def validate_gpu_names(profile: str, names: list[str]) -> None:
    """Do not accidentally accept L40S or another allocation as four L40s."""
    expected = PROFILES[profile].gpu_type
    if len(names) != 4 or any(not re.search(r'\b' + re.escape(expected) + r'\b', name) for name in names):
        raise ValueError(f'{profile} requires exactly four {expected} GPUs; found {names}')

def generation_budget(context: int, input_tokens: int, per_turn: int, reserve: int = 1) -> int:
    if min(context, per_turn) <= 0 or min(input_tokens, reserve) < 0:
        raise ValueError('Invalid token budget')
    return max(0, min(per_turn, context - input_tokens - reserve))
