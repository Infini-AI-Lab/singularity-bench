# Support structured outputs (grammar-constrained decoding) for beam search in vLLM

You are working inside a checkout of **vLLM** at `/app/vllm`, installed **editable with precompiled kernels** —
edits to Python source under `/app/vllm/vllm/` take effect immediately, with **no rebuild**. You can run Python
and `pytest` directly.

## The problem

vLLM's regular generation path supports **structured outputs** (grammar-constrained decoding): you pass a
`StructuredOutputsParams` — for example a JSON schema — and the engine's structured-outputs backend (xgrammar,
guidance, outlines, …) masks the logits at every step so the model can only emit tokens that keep the output
valid. This is how callers force generations to be, say, a JSON object matching a schema.

**Offline beam search does not support this.** `LLM.beam_search(...)` takes a `BeamSearchParams`, but there is no
way to attach a structured-outputs constraint to it, and the beam-search decode loop never consults any grammar.
As a result, beam search emits **free-form text**: if you want a JSON object out of `beam_search`, you have no
way to guarantee (or even nudge) that the returned beams are valid JSON. Users who rely on schema-constrained
decoding therefore cannot use beam search at all.

## Your task

Make offline beam search honor a structured-outputs constraint, matching the behavior users already get from
regular generation:

- `BeamSearchParams` must accept an optional **`structured_outputs`** argument — a `StructuredOutputsParams`
  (the same type regular sampling uses), e.g. `StructuredOutputsParams(json=<schema>)`.
- When it is set, `LLM.beam_search(...)` must constrain decoding with the engine's configured structured-outputs
  backend so that **every returned beam's text conforms to the constraint** (for a JSON schema: every beam is a
  JSON document that validates against that schema).
- When it is **not** set, beam search must behave exactly as before (free-form, no regressions).

The engine is created with a structured-outputs backend already configured (via
`structured_outputs_config=dict(backend="xgrammar", ...)` on the `LLM`), so the backend is available to you at
runtime — you need to drive it from the beam-search loop. Beam search expands and prunes several candidate
sequences per prompt independently, and each candidate is at a different point in the grammar; keeping each
beam's generation valid under the shared schema is the crux of the task.

## Where to look

Offline beam search lives in:

```
/app/vllm/vllm/entrypoints/generate/beam_search/offline.py
```

and the parameter object is `BeamSearchParams` in:

```
/app/vllm/vllm/sampling_params.py
```

The engine's structured-outputs backends (which turn a schema into a per-step token mask) live under
`/app/vllm/vllm/v1/structured_output/`; the regular-generation path is a reference for how a compiled grammar is
advanced and used to mask logits. A small chat model, **TinyLlama-1.1B-Chat-v1.0**, is cached locally so you can
run `LLM.beam_search(...)` end-to-end offline while you iterate.

## How success is decided (runs on GPU)

A hidden test builds an `LLM` for TinyLlama-1.1B with an xgrammar structured-outputs backend, then runs
`llm.beam_search(prompts, BeamSearchParams(beam_width=4, max_tokens=64, structured_outputs=StructuredOutputsParams(json=schema)))`
for a small `{name: string, age: integer}` JSON schema. It asserts that **every** returned beam's text parses as
JSON and validates against the schema. At the parent state `BeamSearchParams` has no `structured_outputs` field
(and the loop applies no grammar), so the test fails; a correct implementation makes every beam valid JSON. The
model runs on a single GPU. There is **no internet access**.
