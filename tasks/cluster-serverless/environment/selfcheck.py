"""Sanity-check your control plane against the fleet before you spend a run on it.

    bash /app/run.sh &
    python3 /app/selfcheck.py

Checks the contract, not the quality: does the plane answer, does it pass receipts through intact, does it
reach a deployed worker and return a complete answer. A plane that fails any of these scores zero no
matter how clever its scheduling is.
"""
from __future__ import annotations

import asyncio
import json
import os
import secrets
import sys

import aiohttp

sys.path.insert(0, os.environ.get("CLUSTER_APP", "/app"))
import receipts  # noqa: E402
import prompts  # noqa: E402

CP = os.environ.get("CLUSTER_CP", "http://127.0.0.1:8000")
FLEET = json.load(open(os.environ.get("CLUSTER_NODES", "/app/fleet.json")))
CAT_META = json.load(open(os.environ.get("CLUSTER_CATALOGUE", "/app/catalogue.json")))
CAT = CAT_META["models"]
KEY = os.environ.get("CLUSTER_RECEIPT_KEY", "dev-key").encode()
FAIL: list[str] = []


def check(name: str, ok: bool, detail: object = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + str(detail) if detail else ''}")
    if not ok:
        FAIL.append(name)


async def main() -> int:
    ctl = FLEET["controller"]
    CTL = f"http://{ctl['host']}:{ctl['port']}"
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as s:
        print("\n[1] your plane is up, and the fleet is reachable")
        try:
            async with s.get(f"{CP}/health") as r:
                check("GET /health on your plane -> 200", r.status == 200, r.status)
        except aiohttp.ClientError as e:
            check("GET /health on your plane -> 200", False, e)
            return 1
        # A plane that is not healthy before the setup window closes scores zero, whatever else it does.
        try:
            async with s.get(f"{CTL}/health") as r:
                check("the controller answers", r.status == 200, r.status)
        except aiohttp.ClientError as e:
            check("the controller answers", False, e)
            return 1

        print("\n[2] the fleet describes itself")
        async with s.get(f"{CTL}/nodes") as r:
            nodes = await r.json() if r.status == 200 else []
        check("GET /nodes reaches every machine", len(nodes) == len(FLEET["nodes"]),
              f"{len(nodes)}/{len(FLEET['nodes'])}")
        check("each node reports its GPUs", all(n.get("gpus") for n in nodes))
        check("and its host RAM", all(n.get("host_ram", {}).get("total_gib", 0) > 0 for n in nodes))
        async with s.get(f"{CTL}/workers") as r:
            live = await r.json() if r.status == 200 else []
        print(f"        {len(live)} engine(s) live across {len(nodes)} node(s)")
        # WHERE THINGS ACTUALLY ARE, not where they were asked to go. Two engines deployed to one node
        # without a `gpus` list both land on card 0; this is how you find out.
        seen: dict[tuple, list] = {}
        for w in live:
            for pl in w.get("placement", []):
                for g in pl.get("gpus", []):
                    seen.setdefault((pl["node"], g["index"]), []).append(w["worker"])
        shared = {k: v for k, v in seen.items() if len(v) > 1}
        if shared:
            print(f"        NOTE: {len(shared)} card(s) hold more than one engine, e.g. "
                  f"{list(shared.items())[0]}. Deliberate is fine; accidental means a missing gpus list.")

        print("\n[3] a request reaches an engine and comes back with a valid receipt")
        model = sorted(CAT, key=lambda m: m["weights_gib"])[0]["id"]
        rid = "selfcheck-1"
        body = {"model": model, "max_tokens": 16, "stream": True,
                "prompt": prompts.with_control(KEY, prompts.tokens(4242, 32), 4),
                "routing": {"request_id": rid, "request_nonce": secrets.token_hex(32),
                            "service_tier": "standard"}}
        out_ids, receipt, finish = [], None, ""
        async with s.post(f"{CP}/v1/completions", json=body) as r:
            check("POST /v1/completions -> 200", r.status == 200, r.status)
            if r.status == 200:
                async for raw in r.content:
                    line = raw.decode(errors="replace").strip()
                    if not line.startswith("data: ") or line.endswith("[DONE]"):
                        continue
                    obj = json.loads(line[6:])
                    if "cluster_receipt" in obj:
                        receipt = obj["cluster_receipt"]
                    else:
                        for ch in obj.get("choices", []):
                            out_ids.extend(ch.get("token_ids") or [])
                            if ch.get("finish_reason"):
                                finish = ch["finish_reason"]
        check("the stream carried a receipt", receipt is not None)
        expect = {"model": model,
                  "prompt_sha256": receipts.sha256_hex(prompts.canon(body["prompt"])),
                  "max_tokens": body["max_tokens"],
                  "text_sha256": receipts.sha256_hex(prompts.canon(out_ids)),
                  "finish_reason": finish, "request_nonce": body["routing"]["request_nonce"]}
        good = bool(receipt) and receipts.check(KEY, expect, receipt)
        check("the receipt verifies against this request", good,
              "" if good else "the model, prompt, token budget or answer is not what the worker signed")
        # A complete answer is one the engine ran to the end of, which is what `finish_reason` reports.
        check("every requested token arrived", good and finish == "length" and len(out_ids) == 4,
              finish or "no finish_reason in the stream")

    print("\nALL PASS" if not FAIL else f"\n{len(FAIL)} FAILURES: {FAIL}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
