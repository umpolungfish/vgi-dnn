#!/usr/bin/env python3
"""Read per-edge and per-bit carrier differences from matched graph evaluations."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from compare_learning_returns import binding


def compare(dataset, runtimes, out, split):
    source = dataset / 'transitions.jsonl'
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    expected = {row['id']: row for row in map(json.loads, source.read_text().splitlines()) if row['split'] == split}
    reports = []
    for runtime in runtimes:
        evaluation = json.loads((runtime / f'graph_rollout_evaluation_{split}.json').read_text())
        assert evaluation['graph_sha256'] == source_sha
        rows = [json.loads(line) for line in (runtime / f'graph_rollout_predictions_{split}.jsonl').open()]
        assert len(rows) == len(expected) and {row['id'] for row in rows} == expected.keys()
        counts = Counter(); bits = [Counter() for _ in range(4)]; changes = []
        for row in rows:
            ref = expected[row['id']]
            assert len(row['edge_values']) == len(ref['edge_values'])
            assert len(row['node_inputs']) == len(ref['node_inputs'])
            assert row['expected'] == ref['return_bits']
            assert row['complete_flow_exact'] == (row['edge_values'] == ref['edge_values'] and row['node_inputs'] == ref['node_inputs'])
            counts['graphs'] += 1; counts['return_exact'] += row['returned'] == ref['return_bits']
            counts['complete_flow_exact'] += row['complete_flow_exact']
            for index, (actual, wanted) in enumerate(zip(row['edge_values'], ref['edge_values'])):
                assert 0 <= actual <= 15
                counts['edges'] += 1; counts['edges_exact'] += actual == wanted
                counts['carrier_bits_exact'] += 4 - (actual ^ wanted).bit_count()
                counts['missing_evidence_bits'] += (wanted & ~actual).bit_count()
                counts['additional_evidence_bits'] += (actual & ~wanted).bit_count()
                for bit, counter in enumerate(bits):
                    a, b = bool(actual & (1 << bit)), bool(wanted & (1 << bit))
                    counter['retained_present' if a and b else 'missing' if b else 'additional' if a else 'retained_absent'] += 1
                if actual != wanted:
                    changes.append({'graph_id': row['id'], 'edge': index, 'expected': wanted,
                                    'actual': actual, 'missing': wanted & ~actual & 15,
                                    'additional': actual & ~wanted & 15})
        reports.append({'binding': binding(runtime, evaluation['module_sha256']), 'counts': dict(counts),
                        'bit_positions': [dict(c) for c in bits], 'edge_differences': changes})
    out.mkdir(parents=True, exist_ok=False)
    report = {'source_sha256': source_sha, 'split': split,
              'scope': 'identical graph sources and references; missing and additional evidence retained separately',
              'readouts': reports}
    (out / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps([{'runtime': row['binding']['runtime'], **row['counts']} for row in reports]))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True); p.add_argument('--runtime', type=Path, action='append', required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--split', choices=('train', 'validation', 'test'), default='validation')
    a = p.parse_args(); compare(a.dataset.resolve(), [p.resolve() for p in a.runtime], a.out.resolve(), a.split)
