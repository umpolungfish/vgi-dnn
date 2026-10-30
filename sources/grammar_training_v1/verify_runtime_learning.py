#!/usr/bin/env python3
"""Verify variable inputs and paired-head SGD in the executing IMASM model."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from distinction_format import words


def verify(root, checkpoint, runtime, examples_path=None):
    torch.set_num_threads(1)
    sys.path.insert(0, str(root / 'bob3ect/reentry_dnn'))
    from dtransformer_torch import DialetheicTransformerTorch, export_to_python
    cfg = json.loads((checkpoint / 'config.json').read_text())
    state = load_file(str(checkpoint / 'model.safetensors'))
    base = DialetheicTransformerTorch(**cfg).double()
    base.load_state_dict(state)
    py = export_to_python(base)
    library = ctypes.CDLL(str((runtime / 'model.so').resolve()))
    for name, arity in [('vgi_input_eval', 3), ('vgi_learn', 5), ('vgi_eval', 1)]:
        method = getattr(library, name)
        method.argtypes = [ctypes.c_long] * arity
        method.restype = ctypes.c_long
    library.vgi_parameter.argtypes = [ctypes.c_long]
    library.vgi_parameter.restype = ctypes.c_double
    library.vgi_value.argtypes = [ctypes.c_long, ctypes.c_long]
    library.vgi_value.restype = ctypes.c_double
    module, vox = runtime / 'model.imasm', root / 'Vox/target/release/vox'
    observations = []

    def vm(symbol, args, expected):
        command = [str(vox), 'run', symbol, str(module), '--args', ','.join(map(str, args))]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=60, check=True)
        output = completed.stdout + completed.stderr
        match = re.search(r'\)\s*=\s*(-?\d+)', output)
        if not match or int(match[1]) != expected:
            raise RuntimeError('VM mismatch: ' + output + ' expected ' + str(expected))
        observations.append({'command': command, 'expected': expected, 'output': output})

    # One new held-out example from every corpus task, plus independent inputs.
    examples = {}
    selection_path = examples_path or checkpoint / 'test_selection.jsonl'
    for line in selection_path.open():
        row = json.loads(line)
        examples.setdefault(row['task'], row)
    input_checks = []
    probes = [(name, [256] + list(row['prompt'].encode('utf8'))[-6:] + [257], row['id'])
              for name, row in sorted(examples.items())]
    probes += [('empty_evidence_probe', [256, 78, 0, 0, 0, 0, 0, 257], None),
               ('both_evidence_probe', [256, 66, 3, 3, 3, 3, 3, 257], None)]
    for name, tokens, source_id in probes:
        first, second = words(tokens)
        for mode in (0, 1):
            digest = library.vgi_input_eval(first, second, mode)
            expected = py.forward(tokens, settle=bool(mode))
            flat = [v for matrix in (expected.support, expected.refute) for row in matrix for v in row]
            error = max(abs(library.vgi_value(i, mode)-v) for i, v in enumerate(flat))
            if error > 1e-6:
                raise RuntimeError(f'native variable-input parity: {name}, {mode}, {error}')
            vm('vgi_input_eval', [first, second, mode], digest)
            input_checks.append({'name': name, 'source_example_id': source_id, 'tokens': tokens,
                                 'reentry': bool(mode), 'max_abs_error': error, 'digest': digest,
                                 'settlement': expected.settlement.kind if mode else None})
    vm('vgi_input_eval', [511, 0, 0], -999999999)

    tokens = json.loads((checkpoint / 'execution_input.json').read_text())['tokens']
    first, second = words(tokens)
    steps = 32
    learning_checks = []
    for bits in range(16):
        model = DialetheicTransformerTorch(**cfg).double()
        model.load_state_dict(state)
        features = []
        hook = model.head.register_forward_pre_hook(lambda module, args: features.extend(args))
        with torch.no_grad():
            model(torch.tensor([tokens]))
        hook.remove()
        t, f = (x.detach() for x in features)
        target = torch.tensor([(bits >> k) & 1 for k in range(4)], dtype=torch.double)
        optimizer = torch.optim.SGD(model.head.parameters(), lr=.05)
        losses = []
        for step in range(steps+1):
            support, refute = model.head(t, f)
            selected = torch.stack([support[0,-1,259], refute[0,-1,259],
                                     support[0,-1,260], refute[0,-1,260]])
            loss = F.binary_cross_entropy_with_logits(selected, target)
            losses.append(float(loss.detach()))
            if step < steps:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        before = library.vgi_learn(first, second, bits, steps, 0)/1e6
        after = library.vgi_learn(first, second, bits, steps, 1)/1e6
        if after >= before or abs(before-losses[0]) > 1e-6 or abs(after-losses[-1]) > 1e-6:
            raise RuntimeError(f'loss parity failed at carrier {bits}')
        expected_params = [v for parameter in (model.head.A, model.head.B, model.head.b)
                           for v in parameter.detach().flatten().tolist()]
        error = max(abs(library.vgi_parameter(i)-v) for i,v in enumerate(expected_params))
        if error > 1e-6:
            raise RuntimeError(f'parameter update parity failed: {bits}, {error}')
        digest = library.vgi_learn(first, second, bits, steps, 2)
        predicted = library.vgi_learn(first, second, bits, steps, 4)
        initial_digest = library.vgi_learn(first, second, bits, steps, 5)
        if digest == initial_digest:
            raise RuntimeError('no parameter change')
        learning_checks.append({'target_bits': bits, 'loss_before': before, 'loss_after': after,
                                'parameter_max_abs_error': error, 'parameter_digest': digest,
                                'predicted_bits': predicted, 'initial_parameter_digest': initial_digest})
        if bits in (0, 1, 2, 3):
            vm('vgi_learn', [first, second, bits, steps, 1], int(after*1e6+.1))
            vm('vgi_learn', [first, second, bits, steps, 2], digest)
            vm('vgi_learn', [first, second, bits, steps, 4], predicted)
    report = {'status': 'passed', 'variable_inputs': input_checks, 'learning': learning_checks,
              'vox_observations': observations, 'learning_steps': steps,
              'learning_scope': 'paired head analytic SGD; retained transformer trunk',
              'readout_scale': 1000000, 'model_checkpoint': str(checkpoint),
              'examples': str(selection_path),
              'module_sha256': hashlib.sha256(module.read_bytes()).hexdigest(),
              'remaining': 'full-network IMASM gradients, persisted learned checkpoints, distinction-choice policy'}
    (runtime / 'learning_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': 'passed', 'input_mode_checks': len(input_checks),
                      'learning_carriers': len(learning_checks), 'vox_calls': len(observations),
                      'head_parameter_max_error': max(r['parameter_max_abs_error'] for r in learning_checks),
                      'exact_after_adaptation': sum(r['predicted_bits']==r['target_bits'] for r in learning_checks)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--examples', type=Path)
    args = parser.parse_args()
    verify(args.root.resolve(), args.checkpoint.resolve(), args.runtime.resolve(), args.examples)
