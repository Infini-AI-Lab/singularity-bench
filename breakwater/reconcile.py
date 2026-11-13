"""Mechanical comparison of writable native logs with externally captured bytes."""
import hashlib
import json
from pathlib import Path
from contract import ContractError, digest, read_json, regular, require, verify_seal

# Every model_response_end outcome the author gateway writes (author_gateway.py).
GATEWAY_OUTCOMES = ('completed', 'client_cancelled', 'upstream_error', 'gateway_stopped')


def check_chain(path, *, gateway=False, identity=None, blob_root=None):
    previous = '0' * 64
    rows = []
    with Path(path).open('rb') as stream:
        for sequence, line in enumerate(stream):
            require(len(line) <= 64 * 1024**2 and line.endswith(b'\n'), 'Truncated or oversized journal event')
            row = json.loads(line)
            if identity is not None:
                require(all(row.get(key) == value for key, value in identity.items()), 'Observer trial/stage identity mismatch')
            require(row.get('seq') == sequence and row.get('previous_sha256') == previous, 'Journal reordered or missing events')
            claimed = row.pop('sha256', None)
            encoded = json.dumps(row, sort_keys=True, **({'separators': (',', ':')} if gateway else {})).encode()
            require(hashlib.sha256(encoded).hexdigest() == claimed, 'Journal hash chain mismatch')
            previous = claimed
            row['sha256'] = claimed
            output = row.get('output')
            if blob_root is not None and isinstance(output, dict) and 'path' in output and 'sha256' in output:
                blob = regular(blob_root, output['path'])
                require(digest(blob) == output['sha256'] and blob.stat().st_size == output.get('bytes'),
                        'Journal refers to changed or incomplete capture bytes')
            rows.append(row)
    require(rows, 'Empty external journal')
    result = {'events': len(rows), 'last_sha256': previous}
    if gateway:
        require(rows[0].get('event') == 'gateway_started' and rows[-1].get('event') == 'gateway_stopped'
                and not rows[-1].get('faulted'), 'Gateway lifecycle incomplete')
        # A failed model call is evidence, not lost capture. The author gateway journals
        # each request before forwarding it and each relayed byte before delivering it,
        # and never reflects an upstream error body. Provider errors, rate limits, failed
        # connections, truncated streams and calls cancelled at shutdown are therefore
        # complete records; they are counted below. Lost capture stays fatal: a recorder
        # error, a faulted stop, an unclosed call, a broken chain or an unknown outcome.
        started, ended, outcomes, succeeded = set(), set(), {}, 0
        for row in rows:
            if row.get('event') == 'model_request':
                require(row.get('call_id') not in started, 'Duplicate model call')
                started.add(row['call_id'])
            if row.get('event') == 'model_response_end':
                require(row.get('call_id') in started and row['call_id'] not in ended, 'Unmatched model result')
                status = row.get('status')
                outcome = row.get('outcome') or ('upstream_error' if row.get('upstream_error') else 'completed')
                require(type(status) is int and outcome in GATEWAY_OUTCOMES, 'Unknown model call outcome')
                outcomes[f'{status} {outcome}'] = outcomes.get(f'{status} {outcome}', 0) + 1
                succeeded += 200 <= status < 300 and not row.get('upstream_error')
                ended.add(row['call_id'])
            require(row.get('event') not in ('collector_error',), 'Gateway capture failed')
        require(started and started == ended, 'Missing model requests/results')
        require(succeeded, 'No author model call succeeded')
        result.update(calls=len(started), succeeded=succeeded, outcomes=dict(sorted(outcomes.items())))
    else:
        require(rows[0].get('event') == 'collector_ready' and rows[-1].get('event') == 'collector_stopped'
                and rows[-1].get('complete') is True and not rows[-1].get('gaps'), 'Observer incomplete')
        require(any(row.get('event') == 'exec_attached' and row.get('protected_stream_supervisor') for row in rows),
                'No protected stream supervisor was observed')
        require(not any(row.get('event') in ('collector_error', 'capture_gap') for row in rows), 'External capture gap')
    return result


def check_service_chain(path, blob_root, source):
    """Development/driver gateway journals: complete chain and lifecycle, every call closed.

    Unlike the author gateway, an upstream error here is an ordinary observation the
    harness under test must survive; it is evidence, not a recorder failure.
    """
    previous = '0' * 64
    rows = []
    with Path(path).open('rb') as stream:
        for sequence, line in enumerate(stream):
            require(len(line) <= 64 * 1024**2 and line.endswith(b'\n'), 'Truncated or oversized gateway event')
            row = json.loads(line)
            require(row.get('seq') == sequence and row.get('previous_sha256') == previous,
                    'Gateway journal reordered or missing events')
            require(row.get('source') == source, 'Gateway journal source mismatch')
            claimed = row.pop('sha256', None)
            encoded = json.dumps(row, sort_keys=True, separators=(',', ':')).encode()
            require(hashlib.sha256(encoded).hexdigest() == claimed, 'Gateway journal hash chain mismatch')
            previous = claimed
            row['sha256'] = claimed
            output = row.get('output')
            if isinstance(output, dict):
                blob = regular(blob_root, output['path'])
                require(digest(blob) == output['sha256'] and blob.stat().st_size == output.get('bytes'),
                        'Gateway journal refers to changed or incomplete bytes')
            rows.append(row)
    require(rows and rows[0].get('event') == 'gateway_started' and rows[-1].get('event') == 'gateway_stopped',
            'Gateway lifecycle incomplete')
    started, ended, statuses = set(), set(), {}
    for row in rows[1:-1]:
        event = row.get('event')
        require(event in ('model_request', 'model_response_end', 'request_rejected'), 'Unknown gateway event')
        if event == 'model_request':
            require(row.get('call_id') not in started and isinstance(row.get('output'), dict),
                    'Duplicate or unrecorded model call')
            started.add(row['call_id'])
        elif event == 'model_response_end':
            require(row.get('call_id') in started and row['call_id'] not in ended, 'Unmatched model result')
            ended.add(row['call_id'])
            statuses[row.get('status')] = statuses.get(row.get('status'), 0) + 1
    require(started == ended, 'Model calls without a recorded result')
    return {'events': len(rows), 'last_sha256': previous, 'calls': len(started),
            'statuses': {str(key): value for key, value in sorted(statuses.items(), key=str)},
            'faulted': bool(rows[-1].get('faulted')), 'final_state': rows[-1].get('state')}


def native_calls(path, *, deadline_reached=False):
    """Normalize native Codex/Claude tool events; unknown records remain byte-checked.

    Only object-shaped `item`/`message` records carry tool calls. Any other shape, such
    as a CLI error or permission-denied event whose message is a string, is an unknown
    record: compare() has already required its exact bytes.

    ``deadline_reached`` (host-recorded: the CLI was still running when the authoring
    budget ended and the host stopped it) tolerates exactly what that stop leaves: a
    final line without its newline, which is not parsed, and at most one tool call
    without a result, the one in flight. A result without its call never passes.
    """
    calls, results = {}, {}
    cut_short = False
    with Path(path).open('rb') as stream:
        for line in stream:
            require(len(line) <= 64 * 1024**2, 'Oversized native JSONL event')
            if not line.endswith(b'\n'):
                # Only the last line can lack its newline; it is still compared byte for byte.
                require(deadline_reached, 'Incomplete native JSONL event')
                cut_short = True
                break
            row = json.loads(line)
            require(isinstance(row, dict), 'Native event must be an object')
            item = row.get('item') if isinstance(row.get('item'), dict) else {}
            if item.get('type') == 'command_execution':
                key = item.get('id')
                require(isinstance(key, str) and key, 'Missing native command identity')
                call = {'command': item.get('command')}
                require(key not in calls or calls[key] == call, 'Command changed under one call ID')
                calls[key] = call
                if row.get('type') == 'item.completed':
                    require(key not in results, 'Duplicate native command result')
                    results[key] = {'output': item.get('aggregated_output'), 'exit_code': item.get('exit_code')}
            message = row.get('message') if isinstance(row.get('message'), dict) else {}
            content = message.get('content')
            for block in content if isinstance(content, list) else []:
                if not isinstance(block, dict):
                    continue
                if block.get('type') == 'tool_use':
                    key = block.get('id')
                    require(isinstance(key, str) and key not in calls, 'Duplicate/missing tool call ID')
                    calls[key] = {'name': block.get('name'), 'input': block.get('input')}
                elif block.get('type') == 'tool_result':
                    key = block.get('tool_use_id')
                    require(isinstance(key, str) and key not in results, 'Duplicate/missing tool result ID')
                    results[key] = {'content': block.get('content'), 'is_error': block.get('is_error', False)}
    in_flight = sorted(set(calls) - set(results))
    require(calls and set(results) <= set(calls) and (not in_flight or (deadline_reached and len(in_flight) == 1)),
            'Native tool call/result coverage is incomplete')
    value = {'calls': calls, 'results': results}
    if deadline_reached:
        value.update(in_flight=in_flight, final_line_cut_short=cut_short)
    return value


def compare(local, protected, *, deadline_reached=False):
    # Exact bytes catch deletions, additions, reordered events and altered output;
    # normalized call/result comparison supplies inspectable mechanical detail.
    def signature(path):
        value = hashlib.sha256()
        size = 0
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024**2), b''):
                value.update(chunk)
                size += len(chunk)
        return size, value.hexdigest()
    require(signature(local) == signature(protected), 'Agent-local native log differs from the external capture')
    left = native_calls(local, deadline_reached=deadline_reached)
    right = native_calls(protected, deadline_reached=deadline_reached)
    require(left == right, 'Tool arguments, outputs or exit statuses differ')
    report = {'status': 'PASS', 'bytes': signature(protected)[0], 'sha256': signature(protected)[1],
              'tool_calls_compared': len(right['calls']), 'tool_results_compared': len(right['results'])}
    if deadline_reached:
        report.update(deadline_reached=True, in_flight_calls=right['in_flight'],
                      final_line_cut_short=right['final_line_cut_short'])
    return report


def reconcile(evidence, authority, local_root, options=None):
    """options: author_gateway (bool), service_gateways ({folder: source}), native ('jsonl'|'streams'),
    agent_deadline_reached (bool, only from a launch record the verifier accepted)."""
    options = options or {}
    manifest = verify_seal(evidence, authority)
    identity = manifest['identity']
    require(manifest.get('finalized') is True and manifest.get('stage_order') == ['agent', 'intermediate'],
            'External evidence was not finalized after both execution stages')
    report = {'protocol': manifest['protocol'], 'identity': identity, 'status': 'PASS', 'observers': {}}
    for stage in ('agent', 'intermediate'):
        ready = read_json(regular(evidence, f'{stage}/ready.json'))
        require(ready.get('ready') is True and ready.get('protocol') == 'external-evidence-v3', 'Recorder protocol/readiness mismatch')
        coverage = read_json(regular(evidence, f'{stage}/coverage.json'))
        require(coverage.get('protocol') == 'external-evidence-v3', 'Recorder coverage protocol mismatch')
        require(coverage.get('complete') is True and not coverage.get('gaps'), f'{stage} recorder incomplete')
        chain = check_chain(regular(evidence, f'{stage}/observer.jsonl'), identity={**identity, 'stage': stage}, blob_root=Path(evidence) / stage)
        require(coverage.get('last_event_sha256') == chain['last_sha256'], 'Coverage anchor mismatch')
        report['observers'][stage] = chain
    if options.get('author_gateway', True):
        report['gateway'] = check_chain(regular(evidence, 'gateway/requests.jsonl'), gateway=True,
                                        blob_root=Path(evidence) / 'gateway')
    for folder, source in sorted((options.get('service_gateways') or {}).items()):
        report[folder] = check_service_chain(regular(evidence, f'{folder}/requests.jsonl'), Path(evidence) / folder, source)
    if options.get('native', 'jsonl') == 'jsonl':
        report['native_agreement'] = compare(regular(local_root, 'native.jsonl'), regular(evidence, 'agent-native.jsonl'),
                                             deadline_reached=options.get('agent_deadline_reached') is True)
    else:
        # A submitted harness is not a native CLI: its protected stdout is byte-compared.
        stdout_local, stdout_external = regular(local_root, 'native.stdout'), regular(evidence, 'agent-native.stdout')
        require(stdout_local.read_bytes() == stdout_external.read_bytes(), 'Agent-local stdout differs from protected stdout')
        report['native_agreement'] = {'status': 'PASS', 'mode': 'streams', 'bytes': stdout_external.stat().st_size,
                                      'sha256': digest(stdout_external)}
    # stderr is not JSONL; it still has exact independent byte equality.
    require(regular(local_root, 'native.stderr').read_bytes() == regular(evidence, 'agent-native.stderr').read_bytes(),
            'Agent-local stderr differs from protected stderr')
    return report
