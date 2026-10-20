#!/usr/bin/env python3
"""Link saved artifact bytes to identical Lean scaffold fields in ob3ect records."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def link(linked, artifacts, out):
    out.mkdir(parents=True, exist_ok=False)
    scaffolds = defaultdict(list)
    for row in read(linked / 'records.jsonl'):
        value = row['payload'].get('lean_scaffold') if row['kind'] == 'ob3ect' else None
        if isinstance(value, str) and value:
            scaffolds[digest(value.encode('utf8'))].append(row)
    counts = Counter()
    with (out / 'links.jsonl').open('w') as stream:
        for artifact in read(artifacts / 'artifacts.jsonl'):
            if artifact['status'] != 'retained':
                continue
            raw = (artifacts / artifact['blob']).read_bytes()
            assert digest(raw) == artifact['sha256']
            owners = {p['id'] for p in artifact['owners']}
            matches = scaffolds.get(artifact['sha256'], [])
            if matches:
                counts['matched_artifact_paths'] += 1
            for row in matches:
                # Compare the full bytes, not a name, printed status, or hash alone.
                assert row['payload']['lean_scaffold'].encode('utf8') == raw
                relation = {'from': artifact['id'], 'to': row['id'],
                            'relation': 'equal complete saved lean_scaffold bytes',
                            'artifact_path': artifact['path'], 'sha256': artifact['sha256'],
                            'target_source': row['source'], 'target_payload_pointer': '/lean_scaffold',
                            'also_in_sibling_inventory': row['id'] in owners}
                stream.write(json.dumps(relation, ensure_ascii=False) + '\n')
                counts['links'] += 1
                counts['sibling_links' if row['id'] in owners else 'other_location_links'] += 1
    report = {'status': 'passed', 'counts': dict(counts),
              'linked_records_sha256': digest((linked / 'records.jsonl').read_bytes()),
              'artifact_inventory_sha256': digest((artifacts / 'artifacts.jsonl').read_bytes()),
              'links_sha256': digest((out / 'links.jsonl').read_bytes()),
              'comparison': 'full strict UTF-8 scaffold field bytes equal retained artifact bytes',
              'kernel_status': 'no scaffold elaboration performed by this link operation'}
    (out / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('linked', 'artifacts', 'out'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    link(*(getattr(args, name).resolve() for name in ('linked', 'artifacts', 'out')))
