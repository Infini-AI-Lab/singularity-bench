"""Public structured-beam invocation; no assertions or expected generated text."""
import contextlib
import json
from pathlib import Path
import sys


def run_cases(input_path, output_path):
    with contextlib.redirect_stdout(sys.stderr):
        from vllm.entrypoints.llm import LLM
        from vllm.platforms import current_platform
        from vllm.sampling_params import BeamSearchParams, StructuredOutputsParams
    cases = json.loads(Path(input_path).read_text())
    with Path(output_path).open('w') as output:
        for case in cases:
            row = {'example_id': case['example_id']}
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    engine = {'async_scheduling': False, 'max_num_seqs': 1, 'enforce_eager': True}
                    if current_platform.is_rocm():
                        engine['enable_prefix_caching'] = False
                    llm = LLM(model=case['model'], dtype=case['dtype'], max_model_len=case['max_model_len'],
                        structured_outputs_config={'backend': 'xgrammar', 'disable_any_whitespace': True}, **engine)
                    params = BeamSearchParams(beam_width=case['beam_width'], max_tokens=case['max_tokens'],
                        structured_outputs=StructuredOutputsParams(json=case['schema']))
                    results = llm.beam_search([case['prompt']], params)
                    returned = [{'sequences': [{'text': sequence.text, 'token_ids': list(sequence.tokens)}
                        for sequence in result.sequences]} for result in results]
                row.update(status='returned', outputs=returned)
            except BaseException as error:
                row.update(status='raised', error_type=type(error).__name__, error=str(error))
            output.write(json.dumps(row, allow_nan=False) + '\n')
            output.flush()


if __name__ == '__main__':
    run_cases(sys.argv[1], sys.argv[2])
