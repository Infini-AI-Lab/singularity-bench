"""Operator-owned native delegation policy, shared by planning and bootstrap."""
SUPPORTED = frozenset({'claude-code', 'codex'})


def enabled(value, harness):
    if value not in ('0', '1'):
        raise ValueError('BREAKWATER_SUBAGENTS must be 0 or 1')
    if value == '1' and harness not in SUPPORTED:
        raise ValueError(f'Subagents are not supported by the protected {harness} profile')
    return value == '1'
