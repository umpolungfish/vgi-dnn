#!/usr/bin/env python3
"""Measure source-family bridges supplied by retained artifact relations."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from build_training_views import Families, read_rows


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit(source, artifacts, scaffolds, out):
    out.mkdir(parents=True, exist_ok=False)
    inputs = [source / f'{split}.jsonl' for split in ('train', 'validation', 'test')]
    inputs += [artifacts / 'artifacts.jsonl', artifacts / 'links.jsonl', scaffolds / 'links.jsonl']
    bindings = {str(p): sha(p) for p in inputs}
    artifact_receipt = json.loads((artifacts / 'verification.json').read_text())
    scaffold_receipt = json.loads((scaffolds / 'verification.json').read_text())
    for name in ('artifacts.jsonl', 'links.jsonl'):
        assert sha(artifacts / name) == artifact_receipt['outputs'][name]
    assert sha(scaffolds / 'links.jsonl') == scaffold_receipt['links_sha256']
    assert sha(artifacts / 'artifacts.jsonl') == scaffold_receipt['artifact_inventory_sha256']
    assignments = {}
    families = Families()
    for split in ('train', 'validation', 'test'):
        for row in read_rows(source / f'{split}.jsonl'):
            assert row['split'] == split
            assignment = (row['family'], split)
            assert assignments.setdefault(row['source_id'], assignment) == assignment
            families.find(row['family'])
    artifact_rows = {}
    for row in read_rows(artifacts / 'artifacts.jsonl'):
        if row['status'] == 'retained':
            artifact_rows[row['id']] = row
            families.join(row['id'], 'artifact-bytes:' + row['sha256'])
            if row['id'] in assignments:
                families.join(row['id'],assignments[row['id']][0])
    relations = []
    for path in (artifacts / 'links.jsonl', scaffolds / 'links.jsonl'):
        for row in read_rows(path):
            assert row['from'] in artifact_rows
            assert row['sha256'] == artifact_rows[row['from']]['sha256']
            if row['to'] in assignments:
                family, split = assignments[row['to']]
                families.join(row['from'], family)
                relations.append({**row, 'source_family': family, 'source_split': split})
    groups = defaultdict(lambda: {'families': {}, 'artifacts': []})
    for family, split in assignments.values():
        groups[families.find(family)]['families'][family] = split
    for artifact in artifact_rows:
        groups[families.find(artifact)]['artifacts'].append(artifact)
    counts = Counter()
    with (out / 'family_bridges.jsonl').open('w') as stream:
        for root, group in sorted(groups.items()):
            if not group['artifacts'] or not group['families']:
                continue
            splits = set(group['families'].values())
            counts['linked_components'] += 1
            counts['cross_split_components'] += len(splits) > 1
            counts['multi_family_components'] += len(group['families']) > 1
            stream.write(json.dumps({'root': root, **group}, ensure_ascii=False) + '\n')
    with (out / 'relations.jsonl').open('w') as stream:
        for row in relations:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    for path, expected in bindings.items():
        assert sha(Path(path)) == expected
    report = {'status': 'passed', 'source': str(source), 'input_bindings': bindings,
              'counts': dict(counts), 'source_relations': len(relations),
              'family_bridges_sha256': sha(out / 'family_bridges.jsonl'),
              'relation_scope': 'retained byte equality, saved scaffold equality, and indexed source-path continuity',
              'sibling_inventory': 'retained separately; co-location alone does not join families'}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'artifacts', 'scaffolds', 'out'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    audit(*(getattr(args, name).resolve() for name in ('source', 'artifacts', 'scaffolds', 'out')))
