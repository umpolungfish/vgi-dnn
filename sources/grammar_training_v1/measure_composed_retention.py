#!/usr/bin/env python3
"""Measure matched held-out corpus and graph returns across training boundaries."""
import argparse
import ctypes
import json
import sys
from pathlib import Path
import torch
from composed_graph_reference import forward
from optimizer_state import load_model
from retained_conditioning import evaluate
from retained_format import record
from train_cuda_trajectories import sha


def measure(root, checkpoints, runtime, dataset, corpus, out):
    torch.set_num_threads(1)
    sys.path.insert(0, str(root / 'bob3ect/reentry_dnn'))
    from dtransformer_torch import DialetheicTransformerTorch
    graphs = [json.loads(line) for line in (dataset / 'transitions.jsonl').open()]
    graphs = [row for row in graphs if row['split'] == 'validation']
    vocabulary = json.loads((dataset / 'vocabulary.json').read_text())
    library = ctypes.CDLL(str(runtime / 'model.so'))
    oracle = library.ig_flow_gate; oracle.argtypes = [ctypes.c_uint] * 5; oracle.restype = ctypes.c_uint
    # A fixed prefix of held-out records bounds this diagnostic. Each chosen
    # source record is evaluated whole and its identity remains in the receipt.
    examples = {}
    with corpus.open() as handle:
        for index, line in enumerate(handle):
            if index >= 512:
                break
            row = json.loads(line)
            assert row['split'] == 'validation'
            frame = record(row)
            size = len(frame['prefix']) + len(frame['target'])
            if size < examples.get(row['task'], (float('inf'), None))[0]:
                examples[row['task']] = size, frame
    results = []
    for checkpoint in checkpoints:
        model = DialetheicTransformerTorch(**json.loads((checkpoint / 'config.json').read_text())).double()
        model.load_state_dict(load_model(checkpoint / 'model.safetensors'))
        graph_results, corpus_results = [], []
        with torch.no_grad():
            for row in graphs:
                result = forward(model, row, vocabulary, oracle, 2, .25)
                predicted = result['returned']; target = row['return_bits']
                graph_results.append({'id': row['id'], 'expected': target, 'returned': predicted,
                    'exact': predicted == target, 'missing': target & ~predicted & 15,
                    'additional': predicted & ~target & 15, 'loss': float(result['loss']),
                    'terminal_loss': float(result['terminal_loss']), 'local_loss': float(result['local_loss'])})
            for task, (_, row) in sorted(examples.items()):
                result = evaluate(model, row['prefix'], row['target'], context=16)
                corpus_results.append({'task': task, 'id': row['example_id'],
                    'source_id': row['source_id'], 'tokens': len(row['prefix']) + len(row['target']),
                    'labels': result['labels'], 'loss': result['loss']})
        results.append({'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint / 'model.safetensors'),
            'graphs': graph_results, 'corpus': corpus_results,
            'graph_exact': sum(row['exact'] for row in graph_results),
            'graph_mean_loss': sum(row['loss'] for row in graph_results) / len(graph_results),
            'corpus_label_mean_loss': sum(row['loss'] * row['labels'] for row in corpus_results) / sum(row['labels'] for row in corpus_results)})
        print(json.dumps({k:v for k,v in results[-1].items() if k not in ('graphs','corpus')}), flush=True)
    report = {'status': 'measured', 'graph_source_sha256': sha(dataset / 'transitions.jsonl'),
        'corpus_source': str(corpus), 'corpus_source_sha256': sha(corpus), 'split': 'validation',
        'corpus_selection': 'shortest complete record per task in first 512 validation rows',
        'graph_count': len(graphs), 'corpus_count': len(examples), 'results': results}
    out.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'runtime', 'dataset', 'corpus', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    measure(args.root.resolve(), [p.resolve() for p in args.checkpoints], args.runtime.resolve(),
            args.dataset.resolve(), args.corpus.resolve(), args.out.resolve())
