#!/usr/bin/env python3
"""Drive your harness through a few ordinary requests so you can see it work.

    python3 /app/selfcheck.py            scripted model, deterministic, free
    python3 /app/selfcheck.py --live     a real model from the served endpoint

Neither is the graded set and neither produces a score. The scripted run answers "does this work at
all"; the live run answers "what does it do when the model does something I did not plan for", which is
the only question the graded run asks.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE = os.path.join(HERE, "smoke")
HARNESS = os.path.join(HERE, "harness", "agent.py")


def main() -> int:
    argv = sys.argv[1:]
    if "--live" in argv:
        rest = [a for a in argv if a != "--live"]
        return subprocess.call([sys.executable, os.path.join(SMOKE, "live.py"), HARNESS, *rest])
    return subprocess.call([sys.executable, os.path.join(SMOKE, "run_smoke.py"), HARNESS, *argv])


if __name__ == "__main__":
    sys.exit(main())
