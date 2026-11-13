"""Answer-free LAMBADA preparation and deterministic audit/generation sampling."""
import hashlib
import json
from pathlib import Path
import random
import tempfile


def encode(rows):
    return b''.join((json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n').encode('utf-8') for row in rows)


def prompt_rows(private_content):
    rows = []
    for index, line in enumerate(private_content.splitlines()):
        source = json.loads(line)
        text = source.get('text')
        if not isinstance(text, str):
            raise ValueError('LAMBADA row requires text')
        parts = text.rstrip().rsplit(maxsplit=1)
        if len(parts) != 2 or not parts[0].strip():
            raise ValueError('LAMBADA row requires context and a final word')
        # Whitelist fields: full text, target word, labels and arbitrary source
        # metadata are never copied. Match generation_eval.read_examples exactly.
        rows.append({'example_id': f'lambada-{index:06d}', 'prompt': parts[0]})
    if not rows:
        raise ValueError('Empty LAMBADA prompt dataset')
    return rows


def validate(path, private_content):
    rows = prompt_rows(private_content)
    expected = encode(rows)
    if Path(path).is_symlink() or Path(path).read_bytes() != expected:
        raise ValueError('Answer-free LAMBADA prompts do not match the pinned private split')
    return rows


def prepare(path, private_content, *, check=False):
    path = Path(path)
    expected = encode(prompt_rows(private_content))
    if path.is_symlink():
        raise ValueError('LAMBADA prompt output must not be a symlink')
    if path.exists() and path.read_bytes() == expected:
        return
    if check:
        raise ValueError('Prepared answer-free LAMBADA prompts are missing or invalid')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.lambada-prompts-', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(expected)
        temporary.chmod(0o644)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def sample(rows, *, count, seed):
    if not 1 <= count <= len(rows):
        raise ValueError('LAMBADA sample size exceeds the available examples')
    indices = sorted(random.Random(seed).sample(range(len(rows)), count))
    selected = [{'example_id': f'lambada-{index:06d}', 'prompt': rows[source]['prompt']}
                for index, source in enumerate(indices)]
    return selected, {'method': 'uniform_without_replacement', 'seed': seed,
        'num_source_examples': len(rows), 'num_sampled_examples': count,
        'source_example_ids': indices, 'sha256': hashlib.sha256(encode(selected)).hexdigest(),
        'source_prompts_sha256': hashlib.sha256(encode(rows)).hexdigest(),
        'input_format': 'lambada-prompts-v1'}
