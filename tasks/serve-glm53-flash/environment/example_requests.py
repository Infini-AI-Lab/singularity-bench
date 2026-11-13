#!/usr/bin/env python3
"""example_requests.py — EXACTLY how the grader talks to YOUR server.

Run it against your running server to see the request/response format first-hand:
    bash /app/start_server.sh          # bring your server up on :8000
    curl -s localhost:8000/v1/models   # wait until ready
    python3 /app/example_requests.py

CONTRACT (what the grader sends, and what this script sends). Everything is
POST http://localhost:8000/v1/completions (OpenAI TEXT-completion, NOT chat), greedy
(temperature 0). TWO request forms occur, and your server must serve both:

  FORM A -- a pre-templated STRING, answered in one JSON body:
      {"model": <any id>, "prompt": <STRING>, "temperature": 0.0,
       "max_tokens": <N>, "stream": false}
      -> {"choices": [{"text": ...}], "usage": {...}}
    The model stops when it stops (EOS), or at max_tokens.

  FORM B -- TOKEN IDS, streamed, generated to an exact length (what /app/simulator.py sends):
      {"model": <any id>, "prompt": [<int>, ...], "temperature": 0.0, "max_tokens": <N>,
       "ignore_eos": true, "skip_special_tokens": false,
       "stream": true, "stream_options": {"include_usage": true}}
      -> server-sent events, one `data: <json>` line each:
           data: {"choices": [{"index": 0, "text": "<piece>"}]}        (as many as you like)
           data: {"choices": [], "usage": {"prompt_tokens": ..., "completion_tokens": ...,
                                          "prompt_tokens_details": {"cached_tokens": ...}}}
           data: [DONE]
    ignore_eos=true means EXACTLY max_tokens tokens come back, EOS or not, and every one of
    them from the same code path as the tokens before EOS (EOS fed back as an ordinary
    token) -- never padding; the brief's rule on this is a gate;
    skip_special_tokens=false means special tokens stay in the text. The ids are the
    whole prompt, chat framing included -- send them to the model as they are.

  * In FORM A, `prompt` is a PRE-TEMPLATED PLAIN STRING — the GLM-5.3-Flash chat template
    is ALREADY APPLIED by the caller. Every such prompt has EXACTLY this shape:

        [gMASK]<sop><|system|>Reasoning Effort: Max<|system|>{system}<|user|>{user}<|assistant|><think>

    Note the TWO <|system|> segments: GLM-5.3's template emits a `Reasoning Effort`
    marker of its own BEFORE the caller's system message. It is part of the prompt and the
    reference answers were produced with it — do NOT strip, reorder or "normalize" it.
    YOUR server tokenizes the string (add_special_tokens=False), decodes greedily
    (temperature 0), and returns choices[0].text — that text is what gets graded.
    GLM is a REASONING model: the prompt ends with `<think>`, so it emits a CoT then
    `</think>` and the answer. Do NOT strip or re-apply the template.

The examples below are SYNTHETIC (NOT the graded prompts) and SHORT — they only show the
FORMATS: single-turn, multi-turn (never sent by the grader; here for template fidelity),
and token ids. The real traffic's
lengths are in /app/simulator.py: prompts of tens of thousands of tokens, up to about
256,000. The only guarantee is (input + output) <= the model's 1,048,576-token context —
do not assume max_tokens is 128/256/1024.
"""
import json, urllib.request

BASE = "http://127.0.0.1:8000"
SYS = "You are a helpful assistant."


# The reasoning-effort marker the GLM-5.3 template emits ahead of everything else.
# Every pre-templated prompt the grader sends is built with effort = "Max".
REASONING_EFFORT = "Max"


def chat(user, prior_turns=()):
    """Build a GLM-5.3-Flash-templated prompt STRING byte-identical to the grader's.

    Equivalent to:
        tok.apply_chat_template(msgs, add_generation_prompt=True,
                                reasoning_effort="max")
    ...which is how the graded workload was actually generated. Verified against the
    tokenizer in the self-check at the bottom of this file."""
    s = f"[gMASK]<sop><|system|>Reasoning Effort: {REASONING_EFFORT}<|system|>{SYS}"
    for u, a in prior_turns:                       # multi-turn -> growing context
        # a PRIOR assistant turn is rendered with an EMPTY think block, then the answer
        s += f"<|user|>{u}<|assistant|><think></think>{a}"
    s += f"<|user|>{user}<|assistant|><think>"
    return s


# Domain-neutral, self-contained example prompts (NOT the graded set) — they only
# demonstrate the request FORMATS. They are short; the real traffic is not.
EXAMPLES = [
    # short single-turn
    {"label": "short", "prompt": chat("In one sentence, what is a hash table?"), "max_tokens": 64},
    # longer single-turn (an open-ended question -> long detailed answer)
    {"label": "long", "prompt": chat(
        "Explain step by step how a modern CPU pipeline executes instructions, "
        "with a concrete example, then summarize the key points."), "max_tokens": 512},
    # multi-turn: a prior Q&A turn already in the context, then a new question
    {"label": "multi-turn", "prompt": chat(
        "Now describe how branch prediction improves on that.",
        prior_turns=[("Briefly, what is instruction pipelining?",
                      "It overlaps the fetch/decode/execute stages of consecutive instructions "
                      "so multiple are in flight at once, raising throughput.")]),
     "max_tokens": 256},
]


def send(prompt, max_tokens):
    payload = json.dumps({
        "model": "/models/GLM-5.3-Flash",   # any id; your server may ignore it
        "prompt": prompt,                          # <-- RAW pre-templated STRING (not tokens)
        "temperature": 0.0,                        # greedy
        "max_tokens": max_tokens,
        "stream": False,
    }).encode()
    req = urllib.request.Request(BASE + "/v1/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["choices"][0]["text"]   # <-- what the grader reads


def send_ids_streaming(ids, max_tokens):
    """FORM B: token ids in, server-sent events out. Returns (text, usage)."""
    payload = json.dumps({
        "model": "/models/GLM-5.3-Flash",
        "prompt": ids,                             # <-- TOKEN IDS, chat framing included
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "ignore_eos": True,                        # exactly max_tokens tokens must come back
        "skip_special_tokens": False,              # special tokens stay in the text
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(BASE + "/v1/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    text, usage = "", None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:                              # one SSE line at a time
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            event = json.loads(line[5:])
            for choice in event.get("choices") or []:
                text += choice.get("text") or ""
            usage = event.get("usage") or usage
    return text, usage


def _selfcheck():
    """Prove chat() matches the model's own tokenizer -- single-turn AND multi-turn -- so
    this file cannot drift from what the grader sends. Skipped if transformers or the
    model is unavailable."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("/models/GLM-5.3-Flash")
        u1, a1, u2 = "Briefly, what is pipelining?", "It overlaps instruction stages.", "And branch prediction?"
        for label, mine, msgs in (
            ("single-turn", chat(u2), [("user", u2)]),
            ("multi-turn ", chat(u2, prior_turns=[(u1, a1)]), [("user", u1), ("assistant", a1), ("user", u2)]),
        ):
            theirs = tok.apply_chat_template(
                [{"role": "system", "content": SYS}] + [{"role": r, "content": c} for r, c in msgs],
                tokenize=False, add_generation_prompt=True,
                reasoning_effort=REASONING_EFFORT.lower())
            print(f"[selfcheck] {label} chat() matches tokenizer template: {mine == theirs}")
            if mine != theirs:
                print(f"[selfcheck]   mine      = {mine!r}")
                print(f"[selfcheck]   tokenizer = {theirs!r}")
    except Exception as e:
        print(f"[selfcheck] skipped ({type(e).__name__}: {e})")


if __name__ == "__main__":
    _selfcheck()
    for ex in EXAMPLES:
        print(f"\n===== {ex['label']}  (prompt {len(ex['prompt'])} chars, max_tokens={ex['max_tokens']}) =====")
        print("--- PROMPT (sent verbatim as the `prompt` string) ---")
        print(ex["prompt"])
        try:
            print("--- RESPONSE choices[0].text ---")
            print(send(ex["prompt"], ex["max_tokens"]))
        except Exception as e:
            print(f"request FAILED: {e}  (is your server up on :8000? `bash /app/start_server.sh`)")

    # FORM B: the same short prompt as token ids, streamed, generated to an exact length.
    print("\n===== token ids + streaming + ignore_eos  (FORM B, max_tokens=48) =====")
    try:
        from transformers import AutoTokenizer
        ids = AutoTokenizer.from_pretrained("/models/GLM-5.3-Flash")(
            EXAMPLES[0]["prompt"], add_special_tokens=False).input_ids
        text, usage = send_ids_streaming(ids, 48)
        print(f"--- {len(ids)} prompt token ids sent; streamed text ---")
        print(text)
        print(f"--- usage: {usage}")
        n = (usage or {}).get("completion_tokens")
        print(f"[check] completion_tokens == max_tokens (ignore_eos honoured): {n == 48}")
        print(f"[check] usage reports prompt_tokens_details.cached_tokens: "
              f"{((usage or {}).get('prompt_tokens_details') or {}).get('cached_tokens') is not None}")
    except Exception as e:
        print(f"request FAILED: {type(e).__name__}: {e}")
