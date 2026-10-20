#!/usr/bin/env python3
"""Compare the same checking queries and graph returns across two checkpoints."""
import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired(before, after, filename):
    def read(directory):
        rows = [json.loads(line) for line in (directory / filename).open()]
        indexed = {row['id']: row for row in rows}
        if len(indexed) != len(rows):
            raise ValueError('duplicate evaluation identity: ' + str(directory))
        return indexed
    left, right = read(before), read(after)
    if left.keys() != right.keys():
        raise ValueError('evaluation identities differ')
    return [(left[key], right[key]) for key in left]


def binding(runtime, module_sha):
    build = json.loads((runtime / 'build.json').read_text())
    if digest(runtime / 'model.imasm') != module_sha:
        raise ValueError('evaluated module changed: ' + str(runtime))
    checkpoint = Path(build['parent'])
    weight_sha = digest(checkpoint / 'model.safetensors')
    source = runtime / 'full_model.cpp'
    if digest(source) != build['source_sha256']:
        raise ValueError('exported source changed')
    match = re.search(r'static const unsigned char header\[\]=\{([\d,]+)\};', source.read_text())
    if match is None:
        raise ValueError('exported checkpoint header missing')
    header = json.loads(bytes(map(int, match[1].split(','))))
    if weight_sha != header['__metadata__']['parent_sha256']:
        raise ValueError('evaluated parent checkpoint changed')
    return {'runtime': str(runtime), 'checkpoint': str(checkpoint),
            'checkpoint_sha256': weight_sha, 'module_sha256': module_sha}


def compare(before_checking, after_checking, before_graph, after_graph, out):
    checking_reports = [json.loads((p / 'checking_evaluation_validation.json').read_text())
                        for p in (before_checking, after_checking)]
    graph_reports = [json.loads((p / 'graph_rollout_evaluation.json').read_text())
                     for p in (before_graph, after_graph)]
    for reports, source_key in ((checking_reports, 'source_sha256'),
                                (graph_reports, 'graph_sha256')):
        if reports[0][source_key] != reports[1][source_key]:
            raise ValueError('evaluation source changed')
    bound = {}
    for label, checking, graph, cr, gr in zip(
            ('before', 'after'), (before_checking, after_checking),
            (before_graph, after_graph), checking_reports, graph_reports):
        c, g = binding(checking, cr['module_sha256']), binding(graph, gr['module_sha256'])
        if c['checkpoint_sha256'] != g['checkpoint_sha256']:
            raise ValueError('checking and graph evaluations use different checkpoints')
        if not cr['forward_only_parameters_exact'] or not cr['optimizer_clock_unchanged']:
            raise ValueError('checking evaluation changed the learner')
        bound[label] = {'checking': c, 'graph': g}
    changes = []; checking_counts = defaultdict(Counter); graph_counts = defaultdict(Counter)
    for a, b in paired(before_checking, after_checking, 'checking_predictions_validation.jsonl'):
        for key in ('expected', 'proposal', 'source_carrier'):
            if a[key] != b[key]:
                raise ValueError('checking query or reference changed')
        ac, bc = a['prediction'] == a['expected'], b['prediction'] == b['expected']
        ar, br = a['returned'] == a['source_carrier'], b['returned'] == b['source_carrier']
        for group in ('all', 'reference_shape:' + str(a['expected'])):
            c = checking_counts[group]; c['queries'] += 1
            c['evidence_gained'] += not ac and bc; c['evidence_lost'] += ac and not bc
            c['carrier_gained'] += not ar and br; c['carrier_lost'] += ar and not br
            c['evidence_exact_before'] += ac; c['evidence_exact_after'] += bc
            c['carrier_exact_before'] += ar; c['carrier_exact_after'] += br
            c['oracle_calls_before'] += a['oracle_calls']; c['oracle_calls_after'] += b['oracle_calls']
        if a != b:
            changes.append({'task': 'checking', 'before': a, 'after': b})
    for a, b in paired(before_graph, after_graph, 'graph_rollout_predictions.jsonl'):
        for key in ('expected', 'split', 'family', 'seed_bits'):
            if a[key] != b[key]:
                raise ValueError('graph query or reference changed')
        for group in ('all', 'split:' + a['split'], 'family:' + a['family']):
            c = graph_counts[group]; c['graphs'] += 1
            for name, ac, bc in (
                    ('return', a['returned'] == a['expected'], b['returned'] == b['expected']),
                    ('complete_flow', a['complete_flow_exact'], b['complete_flow_exact'])):
                c[name + '_gained'] += not ac and bc; c[name + '_lost'] += ac and not bc
                c[name + '_exact_before'] += ac; c[name + '_exact_after'] += bc
        if a != b:
            changes.append({'task': 'graph', 'before': a, 'after': b})
    out.mkdir(parents=True, exist_ok=False)
    changed = out / 'changed_returns.jsonl'
    changed.write_text(''.join(json.dumps(row) + '\n' for row in changes))
    report = {'status': 'passed', 'bindings': bound,
              'checking_source_sha256': checking_reports[0]['source_sha256'],
              'graph_source_sha256': graph_reports[0]['graph_sha256'],
              'checking': dict(checking_counts), 'graph': dict(graph_counts),
              'changed_returns_sha256': digest(changed),
              'scope': 'paired forward evaluation; no evaluation examples used for updates'}
    (out / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': 'passed', 'checking': checking_counts['all'],
                      'graph': graph_counts['all']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('before-checking', 'after-checking', 'before-graph', 'after-graph', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    compare(*(getattr(a, name).resolve() for name in
              ('before_checking', 'after_checking', 'before_graph', 'after_graph', 'out')))
