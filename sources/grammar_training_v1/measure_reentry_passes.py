#!/usr/bin/env python3
"""Measure repeated transformations of retained graph state at frozen weights."""
import argparse
import ctypes
import json
import sys
from pathlib import Path

import torch
from composed_graph_reference import forward
from graph_reentry_bank import load, initial
from optimizer_state import load_model
from train_cuda_trajectories import sha


def measure(root, runtime, dataset, out):
    torch.set_num_threads(1)
    sys.path.insert(0, str(root / 'bob3ect/reentry_dnn'))
    from dtransformer_torch import DialetheicTransformerTorch
    build = json.loads((runtime / 'build.json').read_text())
    parent = Path(build['parent'])
    model = DialetheicTransformerTorch(**build['config']).double()
    model.load_state_dict(load_model(parent / 'model.safetensors'))
    rows = [json.loads(s) for s in (dataset / 'transitions.jsonl').open()]
    bank = load(parent, rows, sha(dataset / 'transitions.jsonl'),
                sha(dataset / 'vocabulary.json'), model.d, build['optimizer_initial_clock'])
    selected = [json.loads(s) for s in (runtime / 'training_graphs.jsonl').open()]
    vocabulary = json.loads((dataset / 'vocabulary.json').read_text())
    library = ctypes.CDLL(str(runtime / 'model.so'))
    oracle = library.ig_flow_gate
    oracle.argtypes = [ctypes.c_uint] * 5
    oracle.restype = ctypes.c_uint
    observations = []
    with torch.no_grad():
        for row in selected:
            state = initial(bank, row, model.d, build['optimizer_initial_clock'])
            if state is None:
                continue
            result = {'id': row['id'], 'target': row['return_bits'], 'returns': []}
            for passes in (2, 4, 8):
                for carried in (False, True):
                    value = forward(model, row, vocabulary, oracle, passes, .25,
                                    initial=state if carried else None)
                    result['returns'].append({'passes': passes, 'carried': carried,
                                              'returned': value['returned'],
                                              'terminal_loss': float(value['terminal_loss']),
                                              'loss': float(value['loss'])})
            observations.append(result)
    summary = []
    for passes in (2, 4, 8):
        for carried in (False, True):
            values = [(r, next(v for v in r['returns'] if v['passes'] == passes and v['carried'] == carried))
                      for r in observations]
            summary.append({'passes': passes, 'carried': carried, 'count': len(values),
                            'exact': sum(v['returned'] == r['target'] for r, v in values),
                            'terminal_loss': sum(v['terminal_loss'] for _, v in values) / len(values)})
    report = {'parent_model_sha256': sha(parent / 'model.safetensors'),
              'bank_sha256': sha(parent / 'composed_graph_bank.json'),
              'summary': summary, 'observations': observations,
              'scope': 'frozen pre-stage weights on selected training graphs; repeated transformations of the same input state, no optimizer updates or validation tuning'}
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'runtime', 'dataset', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    measure(*(getattr(args, name).resolve() for name in ('root', 'runtime', 'dataset', 'out')))
