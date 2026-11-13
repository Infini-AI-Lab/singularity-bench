"""Is every copied module importable where it lands?

_kit/ is the source of truth and three directories get copies of it. A copy that is missing one of its
own imports is not a broken file, it is a container that dies at start.

Static: read each destination's Python files, collect their local imports, and check the closure is
present. No execution, so it also works on files that need aiohttp or a GPU.
"""
from __future__ import annotations

import ast
import pathlib
import re
import sys

T = pathlib.Path(__file__).resolve().parent.parent
KIT = T / "_kit"
LOCAL = {p.stem for p in KIT.glob("*.py")}
DESTS = [T / "environment/_kitcopy", T / "environment/fleetsvc/kit", T / "tests/kit"]
FAIL: list[str] = []
docker = (T / "environment/Dockerfile").read_text()
agent_modules = set(re.findall(r"_kitcopy/(\w+)\.py", docker))


def local_imports(path: pathlib.Path) -> set[str]:
    """Every module from _kit/ that this file imports, at top level or inside a function."""
    out = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.add(node.module.split(".")[0])
    return out & LOCAL


for dest in DESTS:
    present = {p.stem for p in dest.glob("*.py")}
    if not present:
        FAIL.append(f"{dest.relative_to(T)} has no Python files")
        continue
    # The agent build explicitly selects public modules from its staging
    # directory. Fleet and grader copy the whole kit.
    shipped = present & agent_modules if dest == T / 'environment/_kitcopy' else present
    for name in sorted(shipped & LOCAL):
        if (dest / f"{name}.py").read_bytes() != (KIT / f"{name}.py").read_bytes():
            FAIL.append(f"{dest.relative_to(T)}/{name}.py differs from _kit; run _dev/sync_kit.sh")
    # Transitive: a module's imports may import further modules.
    need, seen = set(), set()
    todo = list(present)
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        src = dest / f"{name}.py"
        for dep in local_imports(src if src.exists() else KIT / f"{name}.py"):
            need.add(dep)
            todo.append(dep)
    missing = sorted(need - present)
    tag = str(dest.relative_to(T))
    print(f"  {tag:32} {len(present):2} modules, closure {'complete' if not missing else 'MISSING ' + ', '.join(missing)}")
    if missing:
        FAIL.append(f"{tag} is missing {missing}")

# The agent image's COPY list has to carry the closure of whatever it ships -- checked against the file,
# not the directory. The ROOTS ARE READ FROM THE DOCKERFILE, not listed here, so the check follows the
# image when its COPY list changes.
kitcopy = T / "environment/_kitcopy"
roots = sorted(agent_modules)
need, seen, todo = set(), set(), list(roots)
while todo:
    name = todo.pop()
    if name in seen:
        continue
    seen.add(name)
    src = kitcopy / f"{name}.py"
    for dep in local_imports(src if src.exists() else KIT / f"{name}.py"):
        need.add(dep)
        todo.append(dep)
missing = sorted(n for n in need if f"_kitcopy/{n}.py" not in docker)
print(f"  {'environment/Dockerfile -> /app':32} {len(roots)} roots need {len(need)}, "
      f"{'all copied' if not missing else 'NOT COPIED: ' + ', '.join(missing)}")
if missing:
    FAIL.append(f"the agent image does not COPY {missing}")

print("  ALL PASS" if not FAIL else "  " + "\n  ".join(FAIL))
sys.exit(1 if FAIL else 0)
