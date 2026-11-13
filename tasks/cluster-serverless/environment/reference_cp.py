"""A starter plane. It is DELIBERATELY strategy-free.

What it demonstrates is the CONTRACT, in runnable form: the two routes your gateway must expose, the
shape of a `/deploy` call, how to pass a worker's answer back without invalidating it, and coming up
healthy before the setup window closes. Everything a plane is actually scored on -- what to deploy,
where, how much of a card to give it, what to keep resident, what to evict and when -- is absent on
purpose, and the placement below is arbitrary rather than good.

So: it fills the fleet in catalogue order, deploys AWAKE, routes each request to whichever worker holds
its model, and otherwise does nothing. It never sleeps anything, never reacts to load, never reconsiders
a placement, and when a request names a model it has no room for it retries the deploy on every request.
The tail of the catalogue is never deployed at all.

Two things it does that are worth keeping: it deploys CONCURRENTLY across nodes rather than one at a
time, and it names the GPUs it wants -- deploy without a `gpus` list and every engine on a node lands on
card 0.

Replace it, extend it, or throw it away. The only contract is the HTTP surface in the task instruction
and passing the worker's receipt through unmodified.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import os

import aiohttp
from aiohttp import web

FLEET = json.load(open(os.environ.get("CLUSTER_NODES", "/app/fleet.json")))
CATALOGUE = {m["id"]: m for m in json.load(open(os.environ.get("CLUSTER_CATALOGUE",
                                                               "/app/catalogue.json")))["models"]}
CTL = f"http://{FLEET['controller']['host']}:{FLEET['controller']['port']}"
NODES = FLEET["nodes"]
PORT = int(os.environ.get("CLUSTER_CP_PORT", "8000"))
# An arbitrary slice of a card, and an arbitrary number of engines stacked on one. Both are passed
# straight through (`--gpu-memory-utilization`) and nothing on the platform second-guesses them. There
# is nothing considered about either value -- choosing them is part of the job.
UTIL = 0.30
PER_GPU = 3


class Plane:
    def __init__(self) -> None:
        self.by_model: dict[str, dict] = {}      # model -> the /deploy record (url, worker, ...)
        self.locks: dict[str, asyncio.Lock] = {}
        self.rr = itertools.count()

    def lock(self, model: str) -> asyncio.Lock:
        # Per MODEL, not global -- a global lock would serialise every miss behind every other.
        return self.locks.setdefault(model, asyncio.Lock())

    async def deploy(self, sess: aiohttp.ClientSession, model: str, node: str,
                     gpus: list[int]) -> dict | None:
        body = {"model": model, "placement": [{"node": node, "gpus": gpus}],
                "engine_args": ["--gpu-memory-utilization", f"{UTIL:.2f}"]}
        try:
            async with sess.post(f"{CTL}/deploy", json=body,
                                 timeout=aiohttp.ClientTimeout(total=1800)) as r:
                if r.status != 200:
                    print(f"[plane] deploy {model} on {node}: {r.status} {(await r.text())[:160]}",
                          flush=True)
                    return None
                rec = await r.json()
        except Exception as e:                                            # noqa: BLE001
            print(f"[plane] deploy {model} on {node}: {type(e).__name__}", flush=True)
            return None
        self.by_model[model] = rec
        return rec

    async def warm(self, sess: aiohttp.ClientSession) -> None:
        """Fill the fleet in CATALOGUE ORDER, concurrently, naming the cards.

        Catalogue order is not a heuristic; it is the absence of one. Which models to stage, and in what
        order, is the first decision this starter declines to make for you.
        """
        pick = list(CATALOGUE.values())
        slots = [(nd["name"], gpu) for nd in NODES for gpu in range(int(nd.get("n_gpus") or 8))
                 for _ in range(PER_GPU)]
        jobs = [(m["id"], node, [gpu]) for m, (node, gpu) in zip(pick, slots)]
        done = await asyncio.gather(*(self.deploy(sess, mid, nd, g) for mid, nd, g in jobs))
        print(f"[plane] warm: {sum(1 for d in done if d)} of {len(jobs)} engines up", flush=True)

    async def route(self, sess: aiohttp.ClientSession, model: str) -> dict | None:
        rec = self.by_model.get(model)
        if rec is not None:
            return rec
        async with self.lock(model):
            if (rec := self.by_model.get(model)) is not None:
                return rec
            # No memory of refusals: a model the fleet had no room for is asked for again on the very
            # next request, however many times that is. Naive on purpose.
            nd = NODES[next(self.rr) % len(NODES)]
            return await self.deploy(sess, model, nd["name"], [])


PLANE = Plane()


async def completions(request: web.Request) -> web.StreamResponse:
    body = await request.json()
    sess: aiohttp.ClientSession = request.app["session"]
    rec = await PLANE.route(sess, body.get("model", ""))
    if rec is None:
        return web.json_response({"error": f"no worker for {body.get('model')!r}"}, status=503)
    try:
        async with sess.post(f"{rec['url']}/v1/completions", json=body,
                             timeout=aiohttp.ClientTimeout(total=1800)) as up:
            if up.status != 200:
                return web.json_response({"error": (await up.text())[:200]}, status=up.status)
            resp = web.StreamResponse(status=200,
                                      headers={"Content-Type": "text/event-stream",
                                               "Cache-Control": "no-cache"})
            await resp.prepare(request)
            # PASS THE STREAM THROUGH BYTE FOR BYTE. The receipt frame is in here, and it is what proves
            # a real worker produced this answer; rewriting any of it invalidates the whole response.
            async for chunk in up.content.iter_any():
                await resp.write(chunk)
            return resp
    except Exception as e:                                                # noqa: BLE001
        return web.json_response({"error": f"{type(e).__name__}: {e}"}, status=502)


async def on_start(app: web.Application) -> None:
    app["session"] = aiohttp.ClientSession()
    # Deploy in the BACKGROUND so /health answers immediately. The setup window is a fixed length either
    # way, so there is nothing to gain by making the platform wait for us -- and a plane that is not
    # healthy before the window closes scores zero.
    app["warm"] = asyncio.create_task(PLANE.warm(app["session"]))


async def on_stop(app: web.Application) -> None:
    app["warm"].cancel()
    await app["session"].close()


def build() -> web.Application:
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.on_startup.append(on_start)
    app.on_cleanup.append(on_stop)
    app.add_routes([
        web.post("/v1/completions", completions),
        web.get("/health", lambda r: web.json_response({"status": "ok"})),
    ])
    return app


if __name__ == "__main__":
    web.run_app(build(), host="0.0.0.0", port=PORT, access_log=None)
