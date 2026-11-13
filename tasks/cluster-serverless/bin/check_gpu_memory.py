"""Screen every card's reported memory before staging a fleet.

The ten-percent range is a coarse inventory screen, not a certificate of usable
KV capacity. Missing telemetry fails closed.
"""
import math
import sys


def check(text, expected_cards, nominal_gib):
    rows = text.splitlines()
    if len(rows) != expected_cards:
        raise ValueError(f'expected memory telemetry for {expected_cards} GPUs, received {len(rows)}')
    try:
        memory = [float(row.strip()) / 1024 for row in rows]
    except ValueError as exc:
        raise ValueError('GPU memory telemetry is not numeric') from exc
    if not memory or any(not math.isfinite(v) or v <= 0 for v in memory):
        raise ValueError('GPU memory telemetry is missing or invalid')
    if any(not .9 * nominal_gib <= v <= 1.1 * nominal_gib for v in memory):
        raise ValueError(f'GPU memory range {min(memory):.2f}–{max(memory):.2f} GiB '
                         f'disagrees with the {nominal_gib:g}-GiB catalogue')
    return min(memory), max(memory)


if __name__ == '__main__':
    try:
        low, high = check(sys.stdin.read(), int(sys.argv[1]), float(sys.argv[2]))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
    print(f'{low:.1f}' if low == high else f'{low:.1f}-{high:.1f}')
