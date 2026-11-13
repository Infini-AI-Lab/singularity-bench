#!/usr/bin/env python3
"""Apply the task's narrow Megatron compatibility fixes at image-build time."""

from pathlib import Path
import sys


ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/megatron-lm")


def replace(relative_path: str, old: str, new: str) -> None:
    path = ROOT / relative_path
    source = path.read_text()
    if old not in source:
        raise RuntimeError(f"expected source fragment not found in {path}")
    path.write_text(source.replace(old, new, 1))


override_imports = {
    "megatron/core/optimizer/fully_sharded_optimizer.py": (
        "from typing import Callable, List, Optional, override",
        "from typing import Callable, List, Optional\n\n"
        "try:\n"
        "    from typing import override\n"
        "except ImportError:  # Python < 3.12\n"
        "    from typing_extensions import override",
    ),
    "megatron/training/models/gpt.py": (
        "from typing import Any, Callable, ClassVar, Literal, override",
        "from typing import Any, Callable, ClassVar, Literal\n\n"
        "try:\n"
        "    from typing import override\n"
        "except ImportError:  # Python < 3.12\n"
        "    from typing_extensions import override",
    ),
    "megatron/training/models/hybrid.py": (
        "from typing import Any, Callable, ClassVar, Literal, override",
        "from typing import Any, Callable, ClassVar, Literal\n\n"
        "try:\n"
        "    from typing import override\n"
        "except ImportError:  # Python < 3.12\n"
        "    from typing_extensions import override",
    ),
}

for filename, fragments in override_imports.items():
    replace(filename, *fragments)

gdn_file = "megatron/core/models/gpt/experimental_attention_variant_module_specs.py"
replace(
    gdn_file,
    "from megatron.core.ssm.gated_delta_net import "
    "GatedDeltaNet, GatedDeltaNet2, GatedDeltaNetSubmodules",
    "try:\n"
    "    from megatron.core.ssm.gated_delta_net import (\n"
    "        GatedDeltaNet, GatedDeltaNet2, GatedDeltaNetSubmodules\n"
    "    )\n"
    "except (ImportError, ModuleNotFoundError):\n"
    "    GatedDeltaNet = GatedDeltaNet2 = GatedDeltaNetSubmodules = None",
)
replace(
    gdn_file,
    '    """Build module spec for GatedDeltaNet attention."""\n\n'
    "    if backend is None:",
    '    """Build module spec for GatedDeltaNet attention."""\n\n'
    "    if GatedDeltaNet is None:\n"
    '        raise RuntimeError("Gated Delta Net dependencies are not installed")\n'
    "    if backend is None:",
)
