#!/usr/bin/env python3
"""NAIVE but correct OpenAI-compatible server for GLM-5.3-Flash, built with
plain torch + HuggingFace transformers and nothing faster (no vLLM / SGLang /
flashinfer, no custom kernels).

NOTE ON THE MODEL -- facts from its config.json; what to do with them is yours to work
out. GLM-5.3-Flash (`Glm5NextForConditionalGeneration`, model_type `glm5_next`) is a HYBRID
sparse-MoE REASONING model:
  * 45 layers in a HYBRID attention stack -- 34 `linear_attention` layers and 11
    `deepseek_sparse_attention` layers (DSA on every 4th layer: 3, 7, 11, ...). The two
    kinds carry different state: linear attention a fixed-size recurrent state (conv1d +
    decay), DSA a KV that grows with the sequence.
  * MLA-style compression on the DSA layers: kv_lora_rank 512, q_lora_rank 1536,
    qk_nope_head_dim 256, qk_rope_head_dim 0 (no RoPE split), v_head_dim 256, 64 heads.
  * A learned sparse-attention INDEXER selects which keys each query attends to
    (index_topk 2048, 32 index heads x 128 dim).
  * FFN is a sparse MoE: 288 routed experts, top-8 per token, moe_intermediate_size 2048,
    plus 1 always-on shared expert; the first 3 layers are dense (first_k_dense_replace=3).
  * Hyper-connections, and an MTP (multi-token-prediction) layer.
  * Weights are FP8 e4m3 with DYNAMIC activation scaling (~328 GB total).
  * Context length is 1048576.
  * The checkpoint is multimodal (a vision tower at `model.visual.*`); this task only ever
    sends text.

FOUR GPUS. ~328 GB of FP8 weights do not fit on one of them, so this server shards with
`device_map="auto"`. That is PIPELINE-style placement: the layers are split across the
GPUs and only one GPU computes at a time while the others wait.

It is a THINKING model: prompts carry a `Reasoning Effort` marker and end with
`<|assistant|><think>`, so it emits a chain-of-thought before its answer.

This is the STARTER you optimize: the reference for CORRECT behaviour and for the WIRE
FORMAT, not a server sized for the scored traffic. It is deliberately simple: a FIFO
queue feeds a single background worker that forms a STATIC batch, left-pads it, and runs
ONE `model.generate(...)` greedily until every sequence in the batch hits EOS or
max_tokens. There is NO continuous batching (the whole batch waits for its slowest
member), NO paged KV cache, NO PREFIX CACHING (every request re-prefills its whole
prompt, so it always reports 0 cached tokens), NO token streaming
(a streamed response arrives as ONE chunk when generation ends, so time-to-first-token
equals the full latency), NO custom kernels, and NO tensor parallelism. Correct, but slow.

Exposes (port 8000):
  GET  /v1/models        -> {"data":[{"id": MODEL_ID, ...}]}   (readiness probe)
  GET  /health           -> {"status":"ok"}
  POST /v1/completions    OpenAI-style. Request fields honoured:
         prompt               a pre-templated STRING, or a list of TOKEN IDS
         max_tokens           generation cap
         temperature          ignored: decoding is always greedy
         ignore_eos           true -> keep decoding past EOS, EOS fed back as an ordinary
                              token, to exactly max_tokens, by the SAME code path as
                              before EOS. THIS is the reference behaviour (see NEVER_ID
                              below): no padding, no filler. The brief makes it a gate.
         skip_special_tokens  false -> special tokens stay in the returned text
         stream               true -> server-sent events (`data: {...}` lines, then
                              `data: [DONE]`); with stream_options.include_usage the
                              last event before [DONE] carries `usage`
       Response: {"choices":[{"text": ...}], "usage": {"prompt_tokens",
                  "completion_tokens", "prompt_tokens_details": {"cached_tokens"}}}

String prompts arrive PRE-TEMPLATED (the chat template is already applied), so we
tokenize them verbatim (add_special_tokens=False); token-id prompts are used as they
are. Decoding is greedy.
"""
import os, json, time, threading, asyncio, queue, contextlib, socket
import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/GLM-5.3-Flash")
MODEL_ID = os.environ.get("MODEL_ID", "/models/GLM-5.3-Flash")
# NOTE: prefer SERVER_HOST. A bare `HOST` is a very common variable name and some
# toolchains export it as a BUILD TRIPLET (conda sets HOST=x86_64-conda-linux-gnu),
# which is not a bindable address -- uvicorn then dies with "Name or service not known"
# AFTER the model has finished loading. Validate it and fall back rather than crash.
_host = os.environ.get("SERVER_HOST") or os.environ.get("HOST") or "0.0.0.0"
try:
    socket.getaddrinfo(_host, None)
except Exception:
    print(f"[server] HOST={_host!r} is not a resolvable address; using 0.0.0.0", flush=True)
    _host = "0.0.0.0"
HOST = _host
PORT = int(os.environ.get("PORT", "8000"))
MAX_BATCH = int(os.environ.get("MAX_BATCH", "64"))   # static batch size (naive)
# Cap the TOTAL prompt size per static batch, in tokens. Short prompts still pack up to
# MAX_BATCH; long prompts auto-shrink the batch so the static-batch prefill stays within
# GPU memory, and a prompt longer than the cap runs as a batch of one. 24000 is a
# deliberately conservative choice, not a tuned one: this is meant to be a stable floor.
MAX_BATCH_TOKENS = int(os.environ.get("MAX_BATCH_TOKENS", "24000"))
# Generation length is NOT capped by a magic constant. Each request's own max_tokens is
# honoured, bounded only by THIS MODEL'S OWN CONTEXT LENGTH (read from its config below,
# after the model loads). Requests really do ask for thousands of tokens, so a hardcoded
# cap would silently TRUNCATE answers.
# Set MAX_MODEL_TOKENS in the environment only if you deliberately want a smaller bound.
MAX_MODEL_TOKENS = int(os.environ.get("MAX_MODEL_TOKENS", "0"))   # 0 = use the model's context length
# Prefill the prompt in chunks of this many tokens instead of in one forward pass
# (0 = one-shot). Why it is on: a one-shot prefill of a very long prompt can exceed the
# 2^31 elements a 32-bit kernel index can address (e.g. 131,072 tokens x a 17,408-wide MLP
# is 2.28e9 activations) and then dies with "CUDA error: an illegal memory access". With
# PREFILL_CHUNK=64 a 158-token prompt takes the chunked path and is answered correctly,
# identically as a string and as token ids.
PREFILL_CHUNK = int(os.environ.get("PREFILL_CHUNK", "16384"))
DEVICE = "cuda"

print(f"[server] loading {MODEL_PATH} (FP8 e4m3, sharded over all visible GPUs) ...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
# device_map="auto" spreads the ~328 GB of FP8 weights over every visible GPU because
# they cannot fit on one. dtype="auto" keeps the checkpoint's own FP8 quantization
# config rather than silently upcasting (which would not fit).


def _fp8_config():
    """The checkpoint's own FP8 config, plus two skip entries it gets wrong for transformers.

    `quantization_config.modules_to_not_convert` names the forget-gate projections as
    `model.layers.N.self_attn.f_a_proj` / `...f_b_proj`, but in transformers the modules are
    `model.language_model.layers.N.self_attn.forget_gate.f_a_proj` / `f_b_proj`. The entries
    match nothing, so transformers converts those layers to FP8 although the checkpoint
    stores them UNQUANTIZED. MEASURED with transformers 5.17.0: the load report lists
    `...forget_gate.f_{a,b}_proj.weight_scale_inv | MISSING` for all 45 layers ("newly
    initialized"), i.e. 90 weights dequantized with scales that were never trained. Bare
    names match by suffix (quantizers_utils.should_convert_module), which leaves them the
    plain bf16 linears they are; with them added the load report is clean."""
    with open(os.path.join(MODEL_PATH, "config.json")) as fh:
        q = json.load(fh).get("quantization_config") or {}
    if q.get("quant_method") != "fp8":
        return None
    from transformers.utils.quantization_config import FineGrainedFP8Config
    skip = list(q.get("modules_to_not_convert") or []) + ["f_a_proj", "f_b_proj"]
    return FineGrainedFP8Config(activation_scheme=q.get("activation_scheme", "dynamic"),
                                weight_block_size=tuple(q.get("weight_block_size") or (128, 128)),
                                modules_to_not_convert=skip)


def _load_model():
    """transformers has NO text-only causal-LM class for this architecture. MEASURED with
    transformers 5.17.0: AutoModelForCausalLM.from_pretrained raises "Unrecognized
    configuration class Glm5NextConfig for this kind of AutoModel"; the only registered
    head is Glm5NextForConditionalGeneration (AutoModelForImageTextToText), which carries
    the vision tower as well. generate() on text-only input never runs that tower. The
    causal-LM class is tried first so that a release which adds one is picked up."""
    kw = dict(dtype="auto", device_map="auto", low_cpu_mem_usage=True)
    qc = _fp8_config()
    if qc is not None:
        kw["quantization_config"] = qc
    try:
        return AutoModelForCausalLM.from_pretrained(MODEL_PATH, **kw)
    except ValueError as e:
        print(f"[server] no causal-LM class for this config ({str(e)[:80]}...); "
              f"loading the conditional-generation class instead", flush=True)
        return AutoModelForImageTextToText.from_pretrained(MODEL_PATH, **kw)


model = _load_model().eval()
# With a sharded model, inputs must land on the FIRST shard, not a hardcoded "cuda:0".
DEVICE = model.device
print(f"[server] sharded across {torch.cuda.device_count()} GPU(s); input device={DEVICE}", flush=True)
# Tokens that terminate / pad a finished sequence (cut output here so completion
# token counts and decoded text exclude trailing pad/eos).
_eos = model.generation_config.eos_token_id or model.config.eos_token_id
if not isinstance(_eos, (list, tuple)):
    _eos = [_eos]
STOP_IDS = set(int(x) for x in _eos)
STOP_IDS.add(int(tok.pad_token_id))
print("[server] model loaded. stop_ids=", STOP_IDS, flush=True)
# ignore_eos: generate() stops a sequence at any id in `eos_token_id`, and a config with
# eos_token_id=None is silently refilled from the model's defaults. An id that no token
# can ever equal leaves max_new_tokens as the only stopping rule, while EOS is still
# sampled and kept exactly as the model produces it.
_vocab = max([len(tok)] + [int(getattr(c, "vocab_size", 0) or 0)
                           for c in (getattr(model.config, "text_config", None), model.config) if c is not None])
NEVER_ID = _vocab + 1
# The model's OWN maximum context length, straight from its config -- never a guess.
def _model_ctx():
    for cfg in (getattr(model.config, "text_config", None), model.config):
        if cfg is None:
            continue
        for k in ("max_position_embeddings", "model_max_length", "max_seq_len"):
            v = getattr(cfg, k, None)
            if isinstance(v, int) and v > 0:
                return v
    return 1048576


MODEL_CTX = int(os.environ.get("MODEL_CTX", "0")) or _model_ctx()
print(f"[server] model context length = {MODEL_CTX} tokens; per-request max_tokens is "
      f"honoured up to (MODEL_CTX - prompt_len)", flush=True)


# ---- request plumbing ----
_REQS: "queue.Queue" = queue.Queue()   # of _Item


class _Item:
    __slots__ = ("ids", "max_tokens", "ignore_eos", "skip_special", "fut", "loop")

    def __init__(self, ids, max_tokens, ignore_eos, skip_special, fut, loop):
        self.ids = ids                   # prompt token ids
        self.max_tokens = max_tokens
        self.ignore_eos = ignore_eos
        self.skip_special = skip_special
        self.fut = fut
        self.loop = loop


@torch.inference_mode()
def _run_batch(items):
    ignore_eos = items[0].ignore_eos     # a batch never mixes the two (see _worker)
    # Honour what the caller asked for. The only hard bound is the model's context:
    # prompt tokens + generated tokens must stay inside MODEL_CTX.
    want = max(it.max_tokens for it in items)
    if MAX_MODEL_TOKENS > 0:
        want = min(want, MAX_MODEL_TOKENS)
    enc = tok.pad({"input_ids": [it.ids for it in items]}, return_tensors="pt")   # left-padded
    input_ids = enc.input_ids.to(DEVICE)
    attn = enc.attention_mask.to(DEVICE)
    in_len = input_ids.shape[1]
    max_new = max(1, min(want, MODEL_CTX - in_len))
    out = model.generate(
        input_ids=input_ids, attention_mask=attn,
        max_new_tokens=max_new, do_sample=False, num_beams=1,
        pad_token_id=tok.pad_token_id, use_cache=True,
        **({"prefill_chunk_size": PREFILL_CHUNK} if 0 < PREFILL_CHUNK < in_len else {}),
        **({"eos_token_id": [NEVER_ID]} if ignore_eos else {}),
    )
    gen = out[:, in_len:]               # generated tokens only
    for i, it in enumerate(items):
        seq = gen[i].tolist()
        # truncate at the first stop/pad token (unless the request said ignore_eos) and
        # at this request's own max_tokens
        cut = len(seq)
        if not ignore_eos:
            for j, t in enumerate(seq):
                if t in STOP_IDS:
                    cut = j
                    break
        cut = min(cut, it.max_tokens)
        ids = seq[:cut]
        text = tok.decode(ids, skip_special_tokens=it.skip_special)
        it.loop.call_soon_threadsafe(_set, it.fut, (text, len(ids)))


def _set(fut, val):
    if not fut.done():
        fut.set_result(val)


def _worker():
    while True:
        first = _REQS.get()
        batch = [first]
        ntok = len(first.ids)
        while len(batch) < MAX_BATCH:
            try:
                nxt = _REQS.get_nowait()
            except queue.Empty:
                break
            if ntok + len(nxt.ids) > MAX_BATCH_TOKENS or nxt.ignore_eos != first.ignore_eos:
                _REQS.put(nxt)   # too big for this batch, or the other EOS rule; defer it
                break
            batch.append(nxt)
            ntok += len(nxt.ids)
        try:
            _run_batch(batch)
        except Exception as e:  # never wedge the worker
            for it in batch:
                it.loop.call_soon_threadsafe(_set, it.fut, e)
            if "CUDA error" in str(e):
                # An asynchronous CUDA fault (e.g. an illegal memory access) poisons the
                # context: nothing on this process's GPU state works again. Die visibly
                # rather than answer HTTP 500 to everything forever.
                print(f"[server] FATAL CUDA fault, exiting: {e}", flush=True)
                time.sleep(1.0)          # let the error responses above reach their clients
                os._exit(1)
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()


threading.Thread(target=_worker, daemon=True).start()

app = FastAPI()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}]}


@app.post("/v1/completions")
async def completions(request: Request):
    body = await request.json()
    prompt = body["prompt"]
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], (str, list)):
        prompt = prompt[0]               # a batch of one
    if isinstance(prompt, str):
        ids = tok(prompt, add_special_tokens=False).input_ids
    else:
        ids = [int(t) for t in prompt]   # already token ids
    max_tokens = int(body.get("max_tokens", 1024) or 1024)
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    _REQS.put(_Item(ids, max_tokens, bool(body.get("ignore_eos", False)),
                    bool(body.get("skip_special_tokens", True)), fut, loop))
    res = await fut
    if isinstance(res, Exception):       # an honest failure, not an "[error]" completion
        return JSONResponse({"error": {"message": f"{type(res).__name__}: {res}", "type": "server_error"}},
                            status_code=500)
    text, ntok = res
    usage = {"prompt_tokens": len(ids), "completion_tokens": ntok, "total_tokens": len(ids) + ntok,
             "prompt_tokens_details": {"cached_tokens": 0}}      # no prefix cache here
    base = {"id": "cmpl-local", "object": "text_completion", "model": MODEL_ID}
    choice = {"index": 0, "text": text, "finish_reason": "length" if ntok >= max_tokens else "stop"}
    if not body.get("stream"):
        return JSONResponse({**base, "choices": [choice], "usage": usage})

    def sse():
        # NAIVE: generation has already finished, so the whole text goes out as one event.
        yield f"data: {json.dumps({**base, 'choices': [choice]})}\n\n"
        if (body.get("stream_options") or {}).get("include_usage"):
            yield f"data: {json.dumps({**base, 'choices': [], 'usage': usage})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning",
                limit_concurrency=4096, timeout_keep_alive=600)
