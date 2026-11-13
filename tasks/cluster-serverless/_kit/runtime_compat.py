"""Platform-owned compatibility for the immutable serving checkpoints.

All inference uses vLLM's built-in model implementations and native token IDs.
ERNIE's old checkpoint needs its configuration class, which is not the newer
transformers class with the differently named model_type. Only the reviewed
configuration file at the exact locked revision may enable remote config code.
Checkpoint model implementations are never selected by this task.

Tokenizer initialization is skipped except where vLLM 0.28 cannot start the
model without it. On GB300 the Qwen3.5/3.8 and Gemma 4 input processors read
special-token IDs from the tokenizer at startup, and the Mistral 3 and Voxtral
checkpoints refuse to start unless the tokenizer is mistral-common's. Those
checkpoints load their own locked tokenizer files.
Requests still carry native token IDs and the worker still disables
detokenization, so the served tokens are unchanged. Voxtral's startup
profiling of its audio encoder fails in cuFFT on GB300; requests are text
token IDs, so the encoder never runs and that profiling pass is skipped.
"""
from pathlib import Path

import checkpoints

REVIEWED_CONFIGS = {
    'baidu/ERNIE-4.5-VL-424B-A47B-PT': {
        'revision': 'a3d5a8439d67c9a69003a1f7aac0961c6f1bc20d',
        'name': 'configuration_ernie4_5_vl.py',
        'git_blob_sha1': '8d6b33fb86dfea0b548314a83e7ea440e76afb3e',
        'size': 28314,
    },
}

# Graded checkpoints whose processors need the Hugging Face tokenizer
# (Qwen3_5[Moe]ForConditionalGeneration, Gemma4ForConditionalGeneration).
NEEDS_TOKENIZER = {
    'Qwen/Qwen3.5-9B', 'Qwen/Qwen3.5-122B-A10B', 'Qwen/Qwen3.5-397B-A17B', 'Qwen/Qwen3.8-27B',
    'google/gemma-4-31B-it', 'google/gemma-4-26B-A4B-it',
}
# Graded checkpoints that require --tokenizer-mode mistral (the Mistral 3 and Voxtral architectures).
NEEDS_MISTRAL_TOKENIZER = {
    'mistralai/Ministral-3-3B-Instruct-2512', 'mistralai/Ministral-3-8B-Instruct-2512',
    'mistralai/Mistral-Small-3.2-24B-Instruct-2506', 'mistralai/Mistral-Small-4-119B-2603',
    'mistralai/Voxtral-Small-24B-2507',
}
# Graded checkpoints whose multimodal encoder profiling fails on GB300 (torch.stft: CUFFT_INTERNAL_ERROR).
SKIP_MM_PROFILING = {'mistralai/Voxtral-Small-24B-2507'}


def needs_config_code(checkpoint: str, root: Path | None = None) -> bool:
    reviewed = REVIEWED_CONFIGS.get(checkpoint)
    if reviewed is None:
        return False
    locked = checkpoints.entry(checkpoint)
    expected = {k: reviewed[k] for k in ('git_blob_sha1', 'size')}
    if locked['revision'] != reviewed['revision'] or locked['files'].get(reviewed['name']) != expected:
        raise ValueError('custom configuration code changed; review the new revision first')
    if root is not None:
        path = root/reviewed['name']
        if path.stat().st_size != expected['size'] or checkpoints.file_hash(path, expected) != expected['git_blob_sha1']:
            raise ValueError('custom configuration code does not match the reviewed content')
    return True


def engine_flags(checkpoint: str) -> list[str]:
    flags = ['--model-impl', 'vllm']
    if checkpoint in NEEDS_MISTRAL_TOKENIZER:
        flags += ['--tokenizer-mode', 'mistral']
    elif checkpoint not in NEEDS_TOKENIZER:
        flags.append('--skip-tokenizer-init')
    if checkpoint in SKIP_MM_PROFILING:
        flags.append('--skip-mm-profiling')
    if needs_config_code(checkpoint):
        flags.append('--trust-remote-code')
    return flags
