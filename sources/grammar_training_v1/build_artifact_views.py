#!/usr/bin/env python3
"""Add complete artifact text and source-link attestations to an immutable corpus."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from bootstrap import split_for_address
from build_training_views import read_rows


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build(root, source, artifacts, audit, out):
    out.mkdir(parents=True, exist_ok=False)
    report = json.loads((audit / 'report.json').read_text())
    if report['source'] != str(source):
        raise ValueError('artifact families belong to another source revision')
    for path, expected in report['input_bindings'].items():
        if sha(Path(path)) != expected:
            raise ValueError('artifact family input changed: ' + path)
    if sha(audit / 'family_bridges.jsonl') != report['family_bridges_sha256']:
        raise ValueError('artifact family bridge bytes changed')
    receipt = json.loads((artifacts / 'verification.json').read_text())
    for name, expected in receipt['outputs'].items():
        if sha(artifacts / name) != expected:
            raise ValueError('artifact attachment bytes changed: ' + name)
    assignments = {}
    for group in read_rows(audit / 'family_bridges.jsonl'):
        if len(group['families']) != 1:
            raise ValueError('close artifact family bridges before adding views')
        family, split = next(iter(group['families'].items()))
        for artifact in group['artifacts']:
            assignments[artifact] = (family, split)
    inventory = {row['id']: row for row in read_rows(artifacts / 'artifacts.jsonl')}
    counts = Counter()
    source_hashes = {}
    handles = {split: (out / f'{split}.jsonl').open('w') for split in ('train', 'validation', 'test')}
    try:
        for split, handle in handles.items():
            source_hashes[split] = sha(source / f'{split}.jsonl')
            with (source / f'{split}.jsonl').open() as original:
                for line in original:
                    row = json.loads(line)
                    if row['split'] != split:
                        raise ValueError('source split mismatch')
                    handle.write(line)
                    counts['original_examples'] += 1

        def emit(artifact, task, prompt, target, suffix, extra=None):
            family, split = assignments.get(artifact['id'],
                ('artifact-bytes:' + artifact['sha256'], split_for_address('artifact-bytes:' + artifact['sha256'])))
            provenance = {'path': artifact['path'], 'sha256': artifact['sha256'],
                          'bytes': artifact['bytes'], 'blob': str(artifacts / artifact['blob']),
                          'artifact_id': artifact['id'], **(extra or {})}
            row = {'id': artifact['id'] + ':' + task + ':' + suffix, 'source_id': artifact['id'],
                   'family': family, 'split': split, 'task': task, 'prompt': prompt, 'target': target,
                   'source': provenance, 'evidence': 'saved associated artifact bytes',
                   'dataset_revision': 'artifact_views_v1'}
            handles[split].write(json.dumps(row, ensure_ascii=False) + '\n')
            counts[task] += 1

        for artifact in inventory.values():
            if artifact['status'] != 'retained':
                counts['unavailable_artifacts_retained_in_inventory'] += 1
                continue
            raw = (artifacts / artifact['blob']).read_bytes()
            if hashlib.sha256(raw).hexdigest() != artifact['sha256']:
                raise ValueError('artifact blob changed')
            emit(artifact, 'ob3ect_artifact_attestation',
                 'Return the saved associated-artifact record for this source path.\n' + artifact['path'],
                 json.dumps(artifact, ensure_ascii=False), 'record')
        sys.path.insert(0, str(root / 'VGI/sources/GRAMMAR_UNIFIED_RUN'))
        from grammar_corpus import chunks
        seen = set()
        for text in read_rows(artifacts / 'text.jsonl'):
            artifact = inventory[text['artifact_id']]
            if text['artifact_id'] in seen:
                raise ValueError('duplicate text view')
            seen.add(text['artifact_id'])
            raw = (artifacts / artifact['blob']).read_bytes()
            if text['text'].encode('utf8') != raw:
                raise ValueError('artifact text does not return its source bytes')
            parts = list(chunks(text['text']))
            if not parts:
                parts = [(0, 1, '')]
            if ''.join(part for _, _, part in parts) != text['text']:
                raise ValueError('artifact chunks omit or duplicate source text')
            offset = 0
            for ordinal, line, part in parts:
                cut = len(part) // 3
                emit(artifact, 'ob3ect_artifact_continuation',
                     'Continue this source passage.\n' + part[:cut], part[cut:], str(ordinal),
                     {'start_line': line, 'start_character': offset, 'end_character': offset + len(part)})
                offset += len(part)
            counts['complete_text_artifacts'] += 1
    finally:
        for handle in handles.values():
            handle.close()
    result = {'status': 'built_pending_passage_closure', 'counts': dict(counts),
              'source': str(source), 'source_hashes': source_hashes,
              'artifact_inventory_sha256': sha(artifacts / 'artifacts.jsonl'),
              'artifact_audit_sha256': sha(audit / 'report.json'),
              'outputs': {split: sha(out / f'{split}.jsonl') for split in handles},
              'conditioning': 'complete source chunks; all chunks reconstruct each retained UTF-8 file',
              'binary_artifacts': 'original bytes retained; source-link attestation is the textual training view'}
    (out / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'source', 'artifacts', 'audit', 'out'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    build(*(getattr(args, name).resolve() for name in ('root', 'source', 'artifacts', 'audit', 'out')))
