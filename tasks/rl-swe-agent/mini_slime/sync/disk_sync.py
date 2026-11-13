from __future__ import annotations

import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def update_sglang_from_disk(base_url: str, checkpoint_path: str, version: int, timeout: float = 600.0) -> dict:
    if type(version) is not int or version < 0:
        raise ValueError("weight version must be a nonnegative integer")
    payload = {"model_path": str(Path(checkpoint_path).resolve()),
               "abort_all_requests": True, "weight_version": str(version),
               "flush_cache": True, "is_async": False}
    request = Request(
        f"{base_url.rstrip('/')}/update_weights_from_disk",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f"SGLang weight update HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc
    if not isinstance(result, dict) or result.get("success") is not True:
        raise RuntimeError(f"SGLang rejected checkpoint update: {result}")
    result["local_weight_version"] = version
    return result
