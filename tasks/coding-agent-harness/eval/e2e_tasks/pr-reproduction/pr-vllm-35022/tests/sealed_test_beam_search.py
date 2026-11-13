# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sealed FAIL_TO_PASS test for vLLM PR #35022 — structured outputs for beam search.

This is the single test the PR adds to tests/samplers/test_beam_search.py
(`test_beam_search_structured_output`). The other beam-search tests in that file
are pre-existing and unrelated to this feature, so they are omitted here.

The grader ships this copy at grade time; it is removed from the agent workspace.
"""

import json

import jsonschema
import pytest

from vllm.entrypoints.llm import LLM
from vllm.platforms import current_platform
from vllm.sampling_params import BeamSearchParams, StructuredOutputsParams

# Extra engine kwargs needed for numerically deterministic beam search.
# On ROCm, floating-point reductions in attention and GEMM kernels are
# non-associative and sensitive to batch geometry, so we:
#   async_scheduling=False      – deterministic batch composition
#   enforce_eager=True          – no CUDA-graph padding changing effective size
#   enable_prefix_caching=False – avoid prefix-sharing side effects
#   max_num_seqs=1              – fixed batch size across runs
# On other platforms these are not needed and the dict is empty.
EXTRA_ENGINE_KWARGS: dict = (
    dict(
        async_scheduling=False,
        enforce_eager=True,
        enable_prefix_caching=False,
        max_num_seqs=1,
    )
    if current_platform.is_rocm()
    else dict(async_scheduling=False, max_num_seqs=1)
)

BEAM_WIDTHS = [4]
MODELS = ["TinyLlama/TinyLlama-1.1B-Chat-v1.0"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("beam_width", BEAM_WIDTHS)
def test_beam_search_structured_output(
    model: str,
    dtype: str,
    beam_width: int,
) -> None:
    """Ensure beam search with structured output produces valid JSON."""
    json_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name", "age"],
        "additionalProperties": False,
    }

    llm = LLM(
        model=model,
        dtype=dtype,
        max_model_len=512,
        structured_outputs_config=dict(
            backend="xgrammar",
            disable_any_whitespace=True,
        ),
        **(dict(enforce_eager=True) | EXTRA_ENGINE_KWARGS),
    )

    params = BeamSearchParams(
        beam_width=beam_width,
        max_tokens=64,
        structured_outputs=StructuredOutputsParams(json=json_schema),
    )

    prompts = [
        "Generate a JSON object for a person with name and age:",
    ]

    outputs = llm.beam_search(prompts, params)

    assert len(outputs) == len(prompts)
    for output in outputs:
        assert len(output.sequences) > 0
        for seq in output.sequences:
            assert seq.text is not None
            print(f"Full text: {seq.text!r}")
            # seq.text includes the prompt, extract generated JSON.
            gen_start = seq.text.find("{")
            assert gen_start != -1, f"No JSON found in output: {seq.text!r}"
            generated = seq.text[gen_start:]
            generated = generated.replace("</s>", "").strip()
            print(f"Generated JSON: {generated!r}")
            parsed = json.loads(generated)
            jsonschema.validate(instance=parsed, schema=json_schema)
