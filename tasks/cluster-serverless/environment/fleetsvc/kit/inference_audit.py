"""Private, streaming inference evidence for manual authorship review.

Each worker incarnation owns a new journal on the operator's persistent run
mount. Requests are recorded before inference and response frames before they
are exposed to the caller. Unbuffered writes preserve partial evidence across
worker SIGKILL/unload; fsync on orderly shutdown also flushes the journal. This
does not promise durability across storage or host failure between writes.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import time


class AuditUnavailable(RuntimeError):
    """Inference must stop when its audit cannot be recorded."""


class InferenceAudit:
    def __init__(self, directory: str | Path, metadata: dict):
        if not directory:
            raise AuditUnavailable("CLUSTER_AUDIT_DIR is required")
        root = Path(directory)
        if not root.is_dir():
            raise AuditUnavailable("the operator must create the persistent audit directory")
        self.path = root / f"worker-{secrets.token_hex(16)}.jsonl"
        # The launcher creates the containing directory as the operator, mode
        # 0700. Files must remain readable by that operator when Docker's worker
        # runs as root. Neither this mount nor a read API is given to the agent.
        self.fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        self.failed = False
        self.sequence = 0
        try:
            os.fchmod(self.fd, 0o644)  # preserve operator readability even with a restrictive umask
            self.record("worker_started", schema_version=1, **metadata)
        except BaseException:
            os.close(self.fd)
            self.fd = -1
            raise

    def record(self, event: str, **fields) -> None:
        if self.failed or self.fd < 0:
            raise AuditUnavailable("inference audit is unavailable")
        try:
            data = memoryview((json.dumps({"event": event, "seq": self.sequence,
                "time_ns": time.time_ns(), "utc": datetime.now(timezone.utc).isoformat(),
                "monotonic_ns": time.monotonic_ns(), "source": "private-fleet-worker",
                "trust_level": "trusted_fleet_capture", **fields}, separators=(",", ":"),
                ensure_ascii=True) + "\n").encode())
            # A regular-file write can be short (notably on a full filesystem).
            # Never forward a frame whose full record has not been written.
            while data:
                count = os.write(self.fd, data)
                if count <= 0:
                    raise OSError("audit write made no progress")
                data = data[count:]
            self.sequence += 1
        except (OSError, ValueError) as exc:
            self.failed = True
            raise AuditUnavailable("could not persist inference audit") from exc

    def close(self) -> None:
        if self.fd < 0:
            return
        try:
            if not self.failed:
                self.record("worker_stopped")
            os.fsync(self.fd)
        finally:
            os.close(self.fd)
            self.fd = -1
