r"""Remove every agent CLI from a task container before submitted code runs in it.

    python3 -I purge.py --task-container [--preserve-runtimes]

A task image may contain an agent CLI (for example Claude Code with Node). Submitted code sharing the
container could otherwise hand its work to that CLI -- `claude -p "<task>"` pointed at the stage's own model
through a small local shim -- and the evaluation would measure the CLI instead of the submission; a driver
model could also find one on PATH. The author image runs this before installing its one assigned CLI, and
every grid cell runs it before a submitted harness starts.

The list is deliberately broader than what the task images contain: an image that pre-bakes another agent
must not hand it to every submission. What goes is the installation, not just the command: an npm entry
point is one file of a package that `node` can still run, and a Python entry point sits in a virtualenv
whose package can still be imported. The task's own interpreters and their environments are never touched.
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys

# Names are unambiguous on purpose: deleting a same-named tool a task needs would turn a void into a zero.
CLIS = ("claude", "codex", "kimi", "kimi-cli", "grok", "mini", "mini-extra", "mini-swe-agent", "aider",
        "opencode", "gemini", "qwen", "cursor-agent", "openhands", "sweagent")
# No downstream task needs a JavaScript runtime, and with one a submission could still run a JS agent.
RUNTIMES = ("node", "nodejs", "npm", "npx", "corepack", "bun", "deno")
MODULES = ("minisweagent", "kimi_cli", "aider", "openhands", "sweagent")
# Where these tools keep themselves beyond their entry points.
INSTALL_ROOTS = ("/usr/lib/node_modules", "/usr/local/lib/node_modules", "~/.npm-global", "~/.bun",
                 "~/.grok", "~/.claude/local", "~/.local/share/claude",
                 "~/.local/share/uv/tools/mini-swe-agent", "~/.local/share/uv/tools/kimi-cli")
# Searched as well as PATH: a command does not have to be on PATH to be run.
BIN_DIRS = ("/usr/bin", "/usr/local/bin", "/bin", "/opt/*/bin", "~/.local/bin", "~/.npm-global/bin",
            "~/.grok/bin", "~/.bun/bin", "~/.nvm/versions/node/*/bin", "/home/*/.local/bin")

_FIND = ("import importlib.util as u, sys\n"
         "for m in sys.argv[1:]:\n"
         "    try:\n"
         "        s = u.find_spec(m)\n"
         "    except Exception:\n"
         "        s = None\n"
         "    if s:\n"
         "        print(m, (list(s.submodule_search_locations or []) or [s.origin])[0])\n")


def _dirs(path_env: str, bin_dirs) -> list[str]:
    out = []
    for d in path_env.split(os.pathsep) + [g for p in bin_dirs for g in sorted(glob.glob(os.path.expanduser(p)))]:
        if d and d not in out:
            out.append(d)
    return out


def _candidates(name: str, dirs: list[str]) -> list[str]:
    return [os.path.join(d, name) for d in dirs if os.path.lexists(os.path.join(d, name))]


def _enclosing_venv(path: str) -> str | None:
    d = os.path.dirname(path)
    while d != os.path.dirname(d):
        if os.path.isfile(os.path.join(d, "pyvenv.cfg")):
            return d
        d = os.path.dirname(d)
    return None


def _node_package(path: str) -> str | None:
    head, sep, rest = path.rpartition("/node_modules/")
    if not sep:
        return None
    parts = rest.split("/")
    return head + sep + ("/".join(parts[:2]) if parts[0].startswith("@") else parts[0])


def _pythons(dirs: list[str]) -> list[str]:
    found = []
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "python*"))):
            if (re.fullmatch(r"python(\d+(\.\d+)?)?", os.path.basename(p)) and os.path.isfile(p)
                    and os.access(p, os.X_OK) and p not in found):
                found.append(p)
    return found


def _protected(path_env: str) -> set[str]:
    """The interpreter the harness runs under (the adapter runs this script with the same one) and its
    environment. Only that: an agent's own virtualenv has a python too, and must not be shielded by it.
    An agent installed INTO the task's environment is removed as a package, never by deleting the
    environment around it."""
    keep = {"/", "/usr", "/usr/local", "/opt", os.path.expanduser("~"), os.path.realpath(sys.prefix),
            os.path.realpath(sys.base_prefix), os.path.realpath(sys.executable)}
    for name in ("python3", "python"):
        py = shutil.which(name, path=path_env)
        if py:
            keep.add(os.path.realpath(py))
            venv = _enclosing_venv(py)
            if venv:
                keep.add(os.path.realpath(venv))
    return keep


def _remove(path: str, keep: set[str], log: list[str]) -> None:
    # A symlink is removed as itself; anything else is spared if it IS a protected path.
    if path in keep or (not os.path.islink(path) and os.path.realpath(path) in keep):
        return
    try:
        if os.path.islink(path) or os.path.isfile(path):
            os.unlink(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
        else:
            return
        log.append(f"removed {path}")
    except OSError as e:
        log.append(f"FAILED to remove {path}: {e}")


def _importable(pythons: list[str]) -> list[tuple[str, str, str]]:
    found = []
    for py in pythons:
        try:
            out = subprocess.run([py, "-I", "-c", _FIND, *MODULES], capture_output=True, text=True,
                                 timeout=60).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        for line in out.splitlines():
            module, _, where = line.partition(" ")
            found.append((py, module, where))
    return found


def main(argv=None, path_env=None, bin_dirs=BIN_DIRS, roots=INSTALL_ROOTS) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--task-container" not in argv:
        print("refusing: this deletes agent CLIs and Node system-wide. It is meant for a disposable task "
              "container only (pass --task-container).", file=sys.stderr)
        return 2
    runtimes = () if "--preserve-runtimes" in argv else RUNTIMES
    if not runtimes:
        roots = tuple(p for p in roots if "node_modules" not in p)
    path_env = os.environ.get("PATH", "") if path_env is None else path_env
    dirs = _dirs(path_env, bin_dirs)
    keep = _protected(path_env)
    log: list[str] = []
    for name in CLIS + runtimes:
        for entry in _candidates(name, dirs):
            real = os.path.realpath(entry)
            targets = [entry, real]
            venv = _enclosing_venv(real) or _enclosing_venv(entry)
            if venv and os.path.realpath(venv) not in keep:
                targets.append(venv)
            package = _node_package(real)
            if package:
                targets.append(package)
            for t in targets:
                _remove(t, keep, log)
    for root in roots:
        _remove(os.path.expanduser(root), keep, log)
    for _py, _module, where in _importable(_pythons(dirs)):
        if "site-packages" in where or "dist-packages" in where:
            _remove(where, keep, log)

    left = [p for n in CLIS + runtimes for p in _candidates(n, dirs)]
    left += [f"{py}: import {m} ({w})" for py, m, w in _importable(_pythons(dirs))]
    print("\n".join(log) or "nothing to remove")
    if left:
        print("STILL PRESENT:\n  " + "\n  ".join(left))
        return 1
    print("no agent harness left in this container" + (" (runtimes preserved)" if not runtimes else " or JavaScript runtime"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
