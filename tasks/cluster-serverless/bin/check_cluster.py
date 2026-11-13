#!/usr/bin/env python3
"""Cross-node acceptance check for a fleet started by `cluster_launch.py` (`bin/cluster.sh multinode`).

    python3 bin/check_cluster.py RUN_DIR KIT_DIR

RUN_DIR holds the launched fleet's `fleet.json`, `fleet.public.json` and `catalogue.json`. Checks that
the launcher wrote a usable fleet file and keeps the private/public split, that one controller and one
manager per node answer, and that a model sharded across two nodes deploys, serves a receipt-signed
stream of the marked length, sleeps as a group and unloads idempotently.
"""
import json
import sys
import urllib.error
import urllib.request

RD, KIT = sys.argv[1], sys.argv[2]
sys.path.insert(0, KIT)
import prompts   # noqa: E402
import receipts  # noqa: E402

FAIL = []


def ok(cond, label):
    print(f"  {'ok  ' if cond else 'FAIL'} {label}", flush=True)
    if not cond:
        FAIL.append(label)


def req(method, url, payload=None, timeout=180):
    data = json.dumps(payload).encode() if payload is not None else None
    r = urllib.request.Request(url, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            return e.code, json.loads(body or b"{}")
        except Exception:                                                   # noqa: BLE001
            return e.code, {"raw": body[:200].decode(errors="replace")}


fleet = json.load(open(f"{RD}/fleet.json"))
public = json.load(open(f"{RD}/fleet.public.json"))
cat = json.load(open(f"{RD}/catalogue.json"))
CTL = f"http://{fleet['controller']['host']}:{fleet['controller']['port']}"

print("[1] the launcher wrote a fleet the plane can use")
ok(len(fleet["nodes"]) >= 2, f"the fleet spans at least two nodes (got {len(fleet['nodes'])})")
ports = [n["port"] for n in fleet["nodes"]]
ok(len(set(ports)) == len(ports), "manager ports do not collide across nodes")
ok("controller" in public, "the public view names the controller")
ok(all("speed" not in n for n in public["nodes"]), "the public view does not leak node speeds")
ok("workers" not in public, "and it does not pretend to enumerate workers, which do not exist yet")

print("\n[2] one controller, one manager per node")
st, health = req("GET", f"{CTL}/health")
ok(st == 200 and health.get("nodes") == len(fleet["nodes"]), "the controller answers and sees every node")
st, nodes = req("GET", f"{CTL}/nodes")
ok(st == 200 and len(nodes) == len(fleet["nodes"]), "GET /nodes reaches every manager")
ok(all(n.get("gpus") for n in nodes), "each node reports its GPUs")
ok(all(n.get("host_ram", {}).get("total_gib", 0) > 0 for n in nodes), "and its host RAM")

print("\n[3] a model sharded ACROSS the two machines")
smallest = sorted(cat["models"], key=lambda m: m["weights_gib"])[0]["id"]
names = [n["name"] for n in fleet["nodes"]]
st, d = req("POST", f"{CTL}/deploy",
            {"model": smallest, "engine_args": ["--tensor-parallel-size", "2", "--enable-sleep-mode"],
             "placement": [{"node": names[0], "gpus": [0]}, {"node": names[1], "gpus": [0]}]})
ok(st == 200, f"a two-node deploy succeeds -> {str(d)[:120]}")
if st != 200:
    print(f"\n{len(FAIL)} FAILED")
    sys.exit(1)
ok({p["node"] for p in d["placement"]} == set(names), "placement comes back from BOTH nodes")
ok(d["head"] == names[0], "and the url points at the head node")

print("\n[4] it serves, with a receipt that binds what the worker saw")
st, _ = req("POST", f"{d['url']}/admin/wake", {})
# THE PROTOCOL THE GRADER USES: token IDs on /v1/completions, as in loadgen.py.
WIRE, WANT = 64, 8
sent = prompts.with_control(b"testkey", prompts.tokens(31337, 48), WANT)
body = {"model": smallest, "stream": True, "max_tokens": WIRE, "prompt": sent,
        "routing": {"request_id": "xnode-1"}}
r = urllib.request.Request(f"{d['url']}/v1/completions", data=json.dumps(body).encode(),
                           headers={"Content-Type": "application/json"})
raw = urllib.request.urlopen(r, timeout=120).read().decode()
out_ids, receipt, finish = [], None, ""
for line in raw.splitlines():
    if not line.startswith("data: ") or line.endswith("[DONE]"):
        continue
    ev = json.loads(line[6:])
    for ch in ev.get("choices", []):
        out_ids.extend(ch.get("token_ids") or [])
        if ch.get("finish_reason"):
            finish = ch["finish_reason"]
    if ev.get("cluster_receipt"):
        receipt = ev["cluster_receipt"]
ok(receipt is not None, "the stream carries a receipt")
# Exactly the grader's reconstruction (_kit/loadgen.py).
expect = {"model": smallest,
          "prompt_sha256": receipts.sha256_hex(prompts.canon(sent)),
          "max_tokens": WIRE,
          "text_sha256": receipts.sha256_hex(prompts.canon(out_ids)),
          "finish_reason": finish, "request_nonce": ""}
ok(receipts.check(b"testkey", expect, receipt), "it verifies against the fleet's key")
ok(finish == "length", f"the far shard ran to completion (finish_reason={finish!r})")
# The length rides in the prompt, not in max_tokens: proving the far shard honoured the marker proves
# the control channel survived the cross-node hop, which is the thing this file exists to check.
ok(len(out_ids) == WANT,
   f"the far shard served the marked length ({len(out_ids)} tokens, marker said {WANT})")

print("\n[5] sleep is a group property, and unload is idempotent")
st, sl = req("POST", f"{d['url']}/admin/sleep", {})
ok(st == 200, f"the head sleeps -> {sl}")
st, _ = req("POST", f"{d['url']}/v1/completions", body, timeout=30)
ok(st == 409, f"a sleeping engine refuses to serve (got {st}, want 409)")
st, u = req("POST", f"{CTL}/unload", {"worker": d["worker"]})
ok(st == 200, "unload succeeds")
st, u2 = req("POST", f"{CTL}/unload", {"worker": d["worker"]})
ok(st == 200, "unloading it again still succeeds")
st, ws = req("GET", f"{CTL}/workers")
ok(st == 200 and not ws, "and the fleet is empty again")

print(f"\n  {'ALL PASS' if not FAIL else str(len(FAIL)) + ' FAILED'}")
sys.exit(1 if FAIL else 0)
