"""Slime async binary rewards: symbolic math, option-label science, sandboxed code."""
import asyncio
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import json
import os
import re
import urllib.request

_math_pool = None


def math_pool():
    global _math_pool
    if _math_pool is None:
        # Slime's rollout loop can run in a thread. Math-Verify's signal-based
        # timeouts must execute in the main thread of an independent process.
        _math_pool = ProcessPoolExecutor(max_workers=4,
                                        mp_context=multiprocessing.get_context('spawn'))
    return _math_pool


def final_boxed(text):
    start = text.rfind('\\boxed{')
    if start < 0:
        return None
    start += len('\\boxed{')
    depth = 1
    for end in range(start, len(text)):
        if text[end] == '{':
            depth += 1
        elif text[end] == '}':
            depth -= 1
            if depth == 0:
                return text[start:end].strip()
    return None


def math_reward(response, label):
    from math_verify import LatexExtractionConfig, parse, verify
    answer = final_boxed(response)
    if answer is None:
        return 0.0
    gold = parse('\\boxed{' + str(label) + '}', extraction_config=[LatexExtractionConfig()])
    prediction = parse('\\boxed{' + answer + '}', extraction_config=[LatexExtractionConfig()])
    return float(bool(gold) and bool(prediction) and verify(gold, prediction))


def science_reward(response, label, valid_letters):
    answer = final_boxed(response)
    if answer is None:
        return 0.0
    match = re.fullmatch(r'\\(?:text|mathrm)\{\s*([A-Z])\s*\}|([A-Z])', answer)
    if not match:
        return 0.0
    letter = match.group(1) or match.group(2)
    return float(letter in valid_letters and letter == label)


def extract_code(response):
    blocks = re.findall(r'```(?:python|py)\s*\n(.*?)```', response, re.DOTALL | re.IGNORECASE)
    return blocks[-1].strip() if blocks else None


def code_reward(response, label):
    code = extract_code(response)
    if not code:
        return 0.0
    url = os.environ.get('MIDTRAIN_CODE_SANDBOX_URL')
    if not url:
        raise RuntimeError('MIDTRAIN_CODE_SANDBOX_URL is required; never execute generated code in the grader')
    headers = {'Content-Type': 'application/json'}
    token = os.environ.get('MIDTRAIN_CODE_SANDBOX_TOKEN')
    if token:
        headers['Authorization'] = 'Bearer ' + token
    payload = json.dumps({'code': code, 'label': label}).encode()
    request = urllib.request.Request(url.rstrip('/') + '/score', data=payload, headers=headers)
    with urllib.request.urlopen(request, timeout=600) as result:
        report = json.load(result)
    if report.get('status') != 'ok' or not isinstance(report.get('passed'), bool):
        raise RuntimeError(f'code sandbox infrastructure failed: {report}')
    return float(report['passed'])


async def reward(args, sample, **kwargs):
    domain = sample.metadata['domain']
    if domain == 'math':
        return await asyncio.get_running_loop().run_in_executor(
            math_pool(), math_reward, sample.response or '', sample.label)
    if domain == 'science':
        return science_reward(sample.response or '', sample.label, sample.metadata['valid_letters'])
    if domain == 'code':
        return await asyncio.to_thread(code_reward, sample.response or '', sample.label)
    raise ValueError(f'unknown reward domain: {domain}')
