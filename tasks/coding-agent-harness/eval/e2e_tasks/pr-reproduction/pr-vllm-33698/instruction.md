# Fix pipeline-parallel KV-cache memory validation in vLLM

You are working inside a checkout of **vLLM** at `/app/vllm`, installed **editable with precompiled kernels** —
edits to Python source under `/app/vllm/vllm/` take effect immediately, with **no rebuild**. You can run Python
and `pytest` directly. The fix is **pure-Python KV-cache config arithmetic** — no CUDA kernels are written and no
model weights are loaded (a GPU is present only so vLLM can initialize its device).

## The problem

When vLLM is started with **pipeline parallelism** (`pipeline_parallel_size > 1`), model layers are split across
pipeline stages: each stage runs on its own worker and holds only **its own subset of the layers** (and therefore
only that subset's KV-cache). But vLLM's KV-cache memory validation / config computation does **not** account for
this — it sizes **every** worker as though it had to hold the KV-cache for the **whole** model.

The consequences a user sees:

- Startup fails at KV-cache initialization with a **spurious out-of-memory**: the reported "KV cache needed" is
  roughly **num-pipeline-stages ×** what a single stage actually needs, so a `max_model_len` that comfortably
  fits on each GPU is wrongly rejected (or, under auto-fit, `max_model_len` is silently truncated far below what
  the hardware supports).
- Workers are **coupled** when they should be independent: how much free memory one pipeline stage has ends up
  affecting another stage's KV-cache configuration, even though they hold different layers.

A user filed this after pipeline-parallel serving refused a model that clearly fit across the GPUs.

## Your task

Make KV-cache memory validation and config computation **per-worker**: each pipeline stage must be sized using
**only its own layers** and **only its own available memory**, independently of the other stages.

Concretely, the entry point that computes the KV-cache configuration for the workers,
`get_kv_cache_configs(vllm_config, kv_cache_specs_per_worker, available_memory_per_worker)`, is given a
**per-worker** list of KV-cache specs (the layers that worker actually holds) and a **per-worker** list of
available memory. It must return, for each worker, a config sized for that worker's own layers and memory — so:

- a worker holding one layer's worth of KV-cache, given exactly enough memory for that one layer, is configured
  successfully (no false OOM), and
- giving a *different* worker more memory does **not** change this worker's config.

Non-pipeline-parallel (single-stage) behavior must be unchanged.

## Where to look

The KV-cache configuration logic lives in:

```
/app/vllm/vllm/v1/core/kv_cache_utils.py
```

Look at `get_kv_cache_configs` and the helpers it uses to turn the model-global KV-cache groups and the
available memory into per-worker configs. The bug is in how (or whether) that path narrows the work to the
layers a given worker actually owns.

## How success is decided

A hidden test constructs a **2-stage pipeline** layout — worker 1 holds `layer1`, worker 2 holds `layer2` — and
gives each worker exactly enough memory for its own single layer. It asserts that `get_kv_cache_configs` returns
the correct config for **each** worker (right number of blocks, each tensor sized for that worker's own layer),
in both a **symmetric** case (both workers have the same memory) and an **asymmetric** case (worker 2 has twice
the memory — which must **not** change worker 1's config). At the parent state each worker is sized as if it held
both layers, so the configs are wrong (or the fit is rejected) and the test fails; a correct per-worker fix makes
it pass. No model weights are loaded and there is **no internet access**.
