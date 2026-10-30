#!/usr/bin/env python3
"""Snapshot inventoried ob3ect artifacts as bytes with source and owner links.

No associated program is executed. Sibling location and byte equality are
recorded relations; they do not assert derivation or kernel verification.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def rows(path):
    with path.open() as handle:
        for line in handle:
            yield json.loads(line)


def dump(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def attach(root, linked, documents, out):
    out.mkdir(parents=True, exist_ok=False)
    blobs = out / 'blobs'
    blobs.mkdir()
    owners = defaultdict(list)
    for row in rows(linked / 'records.jsonl'):
        if row['kind'] == 'ob3ect':
            for path in row.get('associated_files', []):
                owners[path].append({'id': row['id'], 'source': row['source']})
    by_path, by_hash = defaultdict(list), defaultdict(list)
    for row in rows(documents / 'files.jsonl'):
        source = row['indexed_source']
        by_path[str(Path(source['path']).absolute())].append(row)
        by_hash[source['sha256']].append(row)
    counts = Counter()
    seen = set()
    with (out / 'artifacts.jsonl').open('w') as inventory, \
         (out / 'text.jsonl').open('w') as text, \
         (out / 'links.jsonl').open('w') as links:
        for relative, parents in sorted(owners.items()):
            path = root / relative
            record = {'id': 'artifact-path:' + sha(relative.encode()),
                      'path': relative, 'owners': parents,
                      'relation': 'inventoried sibling of recorded ob3ect',
                      'execution_status': 'not executed by attachment'}
            # The inventory is untrusted input. Stay inside its source tree and
            # leave symlinks and credential/configuration paths unconsumed.
            allowed = root / 'ob3ect' / 'digital'
            parts = Path(relative).parts
            secret = any(p.lower() in {'.env', '.aws', '.ssh', 'credentials',
                         'credentials.json', 'auth.json', 'token.json'} for p in parts)
            symlink = any(p.is_symlink() for p in (path, *path.parents) if p != root)
            if not path.resolve().is_relative_to(allowed.resolve()) or secret or symlink:
                record['status'] = 'excluded_path_boundary'
            else:
                try:
                    raw = path.read_bytes()
                    digest = sha(raw)
                    blob = blobs / digest
                    if digest not in seen:
                        blob.write_bytes(raw)
                        seen.add(digest)
                    elif blob.read_bytes() != raw:
                        raise ValueError('content hash collision')
                    record.update(status='retained', sha256=digest, bytes=len(raw),
                                  blob='blobs/' + digest, suffix=path.suffix)
                    counts['bytes_by_path'] += len(raw)
                    matches = {}
                    for row in by_path.get(str(path.absolute()), []):
                        matches.setdefault(row['id'], set()).add('same source path')
                    for row in by_hash.get(digest, []):
                        matches.setdefault(row['id'], set()).add('equal indexed source bytes')
                    for target, relations in sorted(matches.items()):
                        dump(links, {'from': record['id'], 'to': target,
                                     'relations': sorted(relations), 'sha256': digest})
                        counts['document_links'] += 1
                    try:
                        content = raw.decode('utf8')
                        if '\0' in content:
                            raise UnicodeError('binary NUL')
                        dump(text, {'artifact_id': record['id'], 'path': relative,
                                    'sha256': digest, 'text': content})
                        record['text_status'] = 'complete strict UTF-8'
                        counts['text_artifacts'] += 1
                    except UnicodeError:
                        record['text_status'] = 'raw bytes retained'
                    if path.read_bytes() != raw:
                        raise ValueError('source changed during attachment: ' + relative)
                except OSError as error:
                    record.update(status='unavailable', error=str(error))
            counts[record['status']] += 1
            counts['owner_links'] += len(parents)
            dump(inventory, record)
    # Independent read-back checks every retained source and every blob, then
    # checks that text views reproduce their full source bytes.
    expected = set(owners)
    retained = {}
    for record in rows(out / 'artifacts.jsonl'):
        expected.remove(record['path'])
        assert record['owners'] == owners[record['path']]
        if record['status'] == 'retained':
            raw = (root / record['path']).read_bytes()
            assert raw == (out / record['blob']).read_bytes()
            assert sha(raw) == record['sha256'] and len(raw) == record['bytes']
            retained[record['id']] = record
    assert not expected
    for record in rows(out / 'text.jsonl'):
        source = retained[record['artifact_id']]
        assert record['text'].encode('utf8') == (out / source['blob']).read_bytes()
    report = {'status': 'passed', 'counts': dict(counts), 'unique_blobs': len(seen),
              'linked_records_sha256': sha((linked / 'records.jsonl').read_bytes()),
              'document_files_sha256': sha((documents / 'files.jsonl').read_bytes()),
              'outputs': {name: sha((out / name).read_bytes()) for name in
                          ('artifacts.jsonl', 'links.jsonl', 'text.jsonl')},
              'verification': 'all retained source bytes, blobs, owner links, and complete UTF-8 views',
              'split_status': 'unassigned; owner and source relations retained for family closure'}
    (out / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'linked', 'documents', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    attach(*(getattr(args, name).resolve() for name in ('root', 'linked', 'documents', 'out')))
