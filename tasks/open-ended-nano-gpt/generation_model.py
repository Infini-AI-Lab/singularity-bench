"""Required agent implementation for the supplied run_generation.sh.

You may replace this module and add supporting code. Keep these three signatures.
The runner loads one checkpoint once, then calls generate for each prompt.
"""


def load_model(model_path: str):
    """Load config and all weights from model_path, with no training or downloads.

Return any model/state object generate needs. Use eval mode. Distributed models
may manage workers internally, within the allocated GPU count.
"""
    raise NotImplementedError("Implement checkpoint loading for your architecture")


def generate(model, prompt_token_ids: list[int], max_new_tokens: int) -> list[int]:
    """Return only newly generated token IDs, never the input prompt.

Use inference mode and CUDA bfloat16 autocast. Autoregressively take argmax over
IDs 0..151668 (tie: lowest ID), conditioning on prompt + your generated tokens.
No sampling, beams, stop-word filters, repetition penalties, target inputs,
weight updates, or external resources. Generate exactly max_new_tokens unless
the model emits end-of-text ID 151643 (include it and stop immediately).
Reset cache/state between prompts. Keep the complete prompt and use causal
attention with correct KV-cache positions. Model-specific workers are allowed.
"""
    raise NotImplementedError("Implement genuine autoregressive greedy generation")


def ppl_logits(model, input_token_ids: list[int]):
    """Return CUDA floating logits [len(input_token_ids), vocabulary >=151669].

Use the SAME checkpoint loaded by load_model. Reset state for every block,
retain every position, and apply causal masking: position i cannot see i+1.
The trusted runner supplies contiguous blocks of at most 2048 inputs and
computes target log probabilities with a 151669-class denominator itself.
Gather vocabulary shards inside this adapter when using distributed models.
No targets, reported losses, or precomputed log probabilities are accepted.
"""
    raise NotImplementedError('Implement causal forward logits for your architecture')
