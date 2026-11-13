# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sealed FAIL_TO_PASS test for vLLM PR #33698 — PP KV-cache sharding memory validation.

Minimal extract of the PR's new behavioral test `test_get_kv_cache_configs_pp_sharding`
from tests/v1/core/test_kv_cache_utils.py, plus the pre-existing `new_kv_cache_spec` helper it
depends on. Pure-CPU: it builds a ModelConfig/VllmConfig and asserts the KV-cache config that
`get_kv_cache_configs` computes for a 2-stage pipeline-parallel layout — no model weights, no GPU.

We keep ONLY the behavioral test (asserts the observable per-worker KV-cache config). The PR also adds
`test_project_kv_cache_groups_to_worker`, which pins the private helper `_project_kv_cache_groups_to_worker`
by name — implementation-coupled, so it is withheld (a correct fix with different internals must still pass).

The grader ships this copy at grade time; it is not in the agent workspace.
"""

import pytest
import torch

from vllm.config import ModelConfig, VllmConfig
from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)


def new_kv_cache_spec(
    block_size=16,
    num_kv_heads=2,
    head_size=64,
    dtype=torch.float32,
    page_size_padded=None,
    sliding_window=None,
    attention_chunk_size=None,
):
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        page_size_padded=page_size_padded,
        sliding_window=sliding_window,
        attention_chunk_size=attention_chunk_size,
    )


@pytest.mark.parametrize(
    "asymmetric_memory",
    [False, True],
    ids=["symmetric", "asymmetric"],
)
def test_get_kv_cache_configs_pp_sharding(asymmetric_memory):
    model_config = ModelConfig(max_model_len=512)
    vllm_config = VllmConfig(model_config=model_config)

    ref_kv_cache_spec = new_kv_cache_spec()
    pp_kv_cache_specs = [
        {"layer1": ref_kv_cache_spec},
        {"layer2": ref_kv_cache_spec},
    ]

    expected_num_blocks = model_config.max_model_len // ref_kv_cache_spec.block_size + 1
    avail_memory = ref_kv_cache_spec.page_size_bytes * expected_num_blocks

    # With per-worker validation, each worker only needs memory for its own
    # layers. Worker 2 having more memory shouldn't affect worker 1's config.
    available_memory = (
        [avail_memory, avail_memory * 2] if asymmetric_memory else [avail_memory] * 2
    )

    kv_cache_configs = get_kv_cache_configs(
        vllm_config,
        pp_kv_cache_specs,
        available_memory,
    )

    assert kv_cache_configs == [
        KVCacheConfig(
            num_blocks=expected_num_blocks,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * expected_num_blocks,
                    shared_by=["layer1"],
                ),
            ],
            kv_cache_groups=[KVCacheGroupSpec(["layer1"], ref_kv_cache_spec)],
        ),
        KVCacheConfig(
            num_blocks=expected_num_blocks,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * expected_num_blocks,
                    shared_by=["layer2"],
                ),
            ],
            kv_cache_groups=[KVCacheGroupSpec(["layer2"], ref_kv_cache_spec)],
        ),
    ]
