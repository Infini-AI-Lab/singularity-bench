# The agent's container. CPU-only on purpose: this task is control-plane engineering, and the GPUs live
# in the remote fleet. Public base, everything installed at build time, offline at run time.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        iptables iproute2 dnsutils curl ca-certificates git build-essential procps jq nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# The agent may build its control plane on any of these; the firewall makes a run-time install impossible.
RUN pip install --no-cache-dir \
        aiohttp==3.14.3 uvloop==0.22.1 httpx==0.28.1 uvicorn==0.34.0 fastapi==0.115.6 \
        orjson==3.10.13 numpy==2.2.1 pulp==2.9.0 sortedcontainers==2.4.0

WORKDIR /app
COPY reference_cp.py run.sh selfcheck.py /app/
COPY control_plane_process.py start_control_plane.sh stop_control_plane.sh /app/
COPY process_safety.md AGENTS.md CLAUDE.md /app/
COPY catalogue-dev.json  /app/catalogue.json
COPY fleet-dev.public.json /app/fleet.json
# Public cases use source partitions and seeds disjoint from the graded ones. The traces ship with the
# task; _dev/world-manifest.json records their hashes.
COPY trace-public-*.jsonl suite-public.json /app/
# `controller.py` ships as the reference for the surface the plane drives -- /deploy, /unload, /nodes,
# /workers -- since the plane has to speak to it correctly to do anything at all.
# No simulator and no physics.py. Characterising the fleet is part of the job: what a deploy, a wake or
# an eviction costs is measurable against the real fleet, which the agent has for the whole window.
COPY _kitcopy/controller.py /app/controller.py
# fleetaddr: where a worker lives (a fleet is not one machine on a cluster).
# engine_args: the reserved-argument list, and what the controller validates a deploy against.
COPY _kitcopy/receipts.py _kitcopy/fleetaddr.py \
     _kitcopy/engine_args.py _kitcopy/prompts.py _kitcopy/service_contract.py /app/
# No score.py. The agent is told WHAT is measured (instruction.md); it is not handed the grader.
#
# `traceunits` ships because `loadgen` needs it to drive a trace at all. It does not leak the deadline:
# the deadline is per request and lives in the trace. The public traces carry their own jittered values;
# the private ones carry different draws, so knowing the rule does not reveal any private request's
# deadline.
COPY _kitcopy/loadgen.py _kitcopy/response_io.py _kitcopy/traceunits.py _kitcopy/receipts.py _kitcopy/looplag.py \
     _kitcopy/fleetaddr.py _kitcopy/prompts.py /app/kit/
RUN cp /app/reference_cp.py /app/control_plane.py && chmod +x /app/*.sh

# Workspace ownership is assigned to the worker by the Breakwater author bootstrap.

# apt is removed last so the firewall cannot be worked around by a stale source list.
RUN rm -rf /etc/apt/sources.list*
