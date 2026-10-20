#!/usr/bin/env python3
"""Render conversational evidence from executing, source-bound distinctions."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path

from distinction_rehearsal import prepare


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def build(records, runtime, out, presence=False):
    library = ctypes.CDLL(str(runtime / 'model.so'))
    pool = prepare([json.loads(s) for s in records.read_text().splitlines()], library)
    rows = []
    for source in pool:
        original = source['record']
        reference = original['reference']
        for action, candidate in enumerate(reference['candidates']):
            bits = (source['target_bits'] >> (2*action)) & 3
            if not bits and not presence:
                continue  # Missing evidence supplies no invented text label.
            value = source['candidate_returns'][action]
            assert value == candidate['return_bits']
            prompt = (f"User: What affirmative and negative evidence is recorded for {candidate['action']}?\n"
                      f"Operation: {reference['work']}; input: {reference['seed_bits']}.\n"
                      f"Affirmative return criterion: {reference['positive_return']}.\n"
                      f"Negative return criterion: {reference['negative_return']}.\n"
                      f"Requested view: {reference['requested_view']}; candidate view: {candidate['view']}.\nAssistant:")
            support = (f"The candidate returns {value}, matching the affirmative return and view criteria."
                       if bits & 1 else None)
            witnesses = []
            if reference['negative_return'] is not None and value == reference['negative_return']:
                witnesses.append(f"its return {value} matches the negative criterion")
            if reference['requested_view'] is not None and candidate['view'] != reference['requested_view']:
                witnesses.append(f"its view {candidate['view']} differs from the requested view {reference['requested_view']}")
            refutation = 'The candidate is refuted because ' + ' and '.join(witnesses) + '.' if bits & 2 else None
            assert not (bits & 2) or witnesses
            rows.append({'kind': 6, 'task': 'paired_language_evidence',
                         'id': f"paired-language:{original['example_id']}:{action}",
                         'example_id': f"paired-language:{original['example_id']}:{action}",
                         'source_id': original['source_id'], 'family': original['family'],
                         'training_split': 'train', 'source': original['source'],
                         'prompt': prompt, 'prefix': [256] + list(prompt.encode()) + [257], 'target': [],
                         'prompt_sha256': digest(prompt), 'support_target': support,
                         'refutation_target': refutation,
                         'support_sha256': digest(support) if support is not None else None,
                         'refutation_sha256': digest(refutation) if refutation is not None else None,
                         'evidence_bits': bits, 'action': action, 'oracle_binding': source})
            if presence:
                rows[-1]['evidence_presence'] = True
    out.mkdir(exist_ok=False, parents=True)
    (out / 'records.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    report = {'records': len(rows), 'both': sum(r['evidence_bits'] == 3 for r in rows),
              'source_records': str(records), 'source_records_sha256': hashlib.sha256(records.read_bytes()).hexdigest(),
              'native_module_sha256': hashlib.sha256((runtime / 'model.so').read_bytes()).hexdigest(),
              'missing_evidence': 'masked; no fabricated refutation or affirmative target',
              'scope': 'oracle-grounded conversational evidence for existing distinction operations'}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('records', 'runtime', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--presence', action='store_true')
    args = parser.parse_args()
    build(*(getattr(args, name).resolve() for name in ('records', 'runtime', 'out')), presence=args.presence)
