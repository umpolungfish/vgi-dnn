#!/usr/bin/env python3
"""Compare all example content and provenance across a split-family revision."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from close_split_families import rows


ASSIGNMENT_FIELDS = {'family', 'split', 'prior_family', 'prior_split', 'family_history', 'dataset_revision'}


def content(row):
    return hashlib.sha256(json.dumps({k: v for k, v in row.items() if k not in ASSIGNMENT_FIELDS},
                                    sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def verify(source, revised, artifact_audit=None):
    expected = {}
    for row in rows(source):
        assert row['id'] not in expected
        history = list(row.get('family_history', []))
        if 'prior_family' in row:
            history.append({'family': row['prior_family'], 'split': row['prior_split']})
        history.append({'family': row['family'], 'split': row['split']})
        expected[row['id']] = (content(row), row['family'], history)
    total = len(expected)
    families, movements = {}, {}
    counts = Counter()
    for row in rows(revised):
        digest, previous, history = expected.pop(row['id'])
        assert content(row) == digest, 'example content or provenance changed'
        assert row['prior_family'] == previous
        assert families.setdefault(row['family'], row['split']) == row['split']
        assert movements.setdefault(previous, (row['family'], row['split'])) == (row['family'], row['split'])
        actual = row['family_history']
        position = 0
        for entry in history:
            # Adjacent repeated observations may share one history entry.
            assert entry in actual[position:], 'prior assignment history lost'
            position = actual.index(entry, position)
        counts[row['split']] += 1
    assert not expected, 'source examples missing'
    checked = 0
    if artifact_audit:
        for line in (artifact_audit / 'family_bridges.jsonl').open():
            group = json.loads(line)
            assignments = {movements[family] for family in group['families']}
            assert len(assignments) == 1, 'artifact relation still crosses revised families'
            checked += 1
    report = {'status': 'passed', 'examples': total, 'counts': dict(counts),
              'checks': ['all IDs retained exactly once', 'all content and source provenance unchanged',
                         'prior assignment history retained', 'each family occupies one split'],
              'artifact_components_checked': checked, 'source': str(source), 'revised': str(revised)}
    (revised / 'preservation_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'revised'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--artifact-audit', type=Path)
    args = parser.parse_args()
    verify(args.source.resolve(), args.revised.resolve(), args.artifact_audit.resolve() if args.artifact_audit else None)
