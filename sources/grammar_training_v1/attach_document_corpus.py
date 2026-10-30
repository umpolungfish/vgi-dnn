#!/usr/bin/env python3
"""Export the existing document/Lean index with checked source bindings.

Retains indexed text even when a source changed. Such rows are explicitly marked
for refresh. Generated copies and exact duplicates stay visible for family review.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3


def file_hash(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def export(database, linked, output):
    database, linked, output = database.resolve(), linked.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    db_hash = file_hash(database)
    con = sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    con.execute('BEGIN')
    names = defaultdict(list)
    with (linked / 'records.jsonl').open() as handle:
        for line in handle:
            row = json.loads(line)
            if row['kind'] == 'catalog_entry':
                name = row['payload'].get('name')
                if isinstance(name, str):
                    names[name].append(row['id'])
    counts, freshness = Counter(), Counter()
    duplicates = defaultdict(list)
    source_ids = {}
    with (output / 'files.jsonl').open('w', encoding='utf8') as files:
        for sql_row in con.execute('SELECT * FROM files ORDER BY id'):
            row = dict(sql_row)
            path = Path(row['path'])
            try:
                current_hash = file_hash(path)
                status = 'matched' if current_hash == row['sha256'] else 'changed'
                error = None
            except OSError as exc:
                current_hash, status, error = None, 'unavailable', str(exc)
            record_id = f'corpus:{db_hash}:file:{row["id"]}'
            source_ids[row['id']] = record_id
            parts = set(path.parts)
            generated = bool(parts & {'.build', '.ltx-build', '__pycache__'})
            record = {'id': record_id, 'kind': row['kind'], 'indexed_source': row,
                      'current_source_sha256': current_hash, 'source_status': status,
                      'source_error': error, 'generated_path': generated,
                      'split': None, 'kernel_status': 'unmeasured'}
            files.write(json.dumps(record, ensure_ascii=False) + '\n')
            counts[row['kind']] += 1
            counts['generated_path'] += generated
            freshness[status] += 1
            if row['sha256']:
                duplicates[row['sha256']].append(record_id)

    with (output / 'links.jsonl').open('w', encoding='utf8') as links:
        for table in ('chunks', 'declarations', 'imports'):
            with (output / f'{table}.jsonl').open('w', encoding='utf8') as target:
                for sql_row in con.execute(f'SELECT * FROM {table} ORDER BY id'):
                    row = dict(sql_row)
                    record_id = f'corpus:{db_hash}:{table}:{row["id"]}'
                    record = {'id': record_id, 'source_id': source_ids[row['file_id']],
                              'payload': row, 'extraction': 'existing_index',
                              'kernel_status': 'unmeasured'}
                    target.write(json.dumps(record, ensure_ascii=False) + '\n')
                    counts[table] += 1
                    if table == 'declarations':
                        for catalog_id in names.get(row['name'], []):
                            links.write(json.dumps({'from': record_id, 'to': catalog_id,
                                                    'relation': 'exact_declaration_name_match'}) + '\n')
                            counts['catalog_declaration_links'] += 1
    con.close()
    if file_hash(database) != db_hash:
        raise RuntimeError('source index changed during export; export needs review')
    duplicate_groups = [group for group in duplicates.values() if len(group) > 1]
    (output / 'duplicates.json').write_text(json.dumps(duplicate_groups, indent=2) + '\n')
    report = {'database': str(database), 'database_sha256': db_hash,
              'linked_records': str(linked / 'records.jsonl'),
              'linked_records_sha256': file_hash(linked / 'records.jsonl'),
              'counts': dict(counts), 'source_status': dict(freshness),
              'duplicate_source_hash_groups': len(duplicate_groups),
              'declaration_status': 'regex extraction; not elaborated proof evidence',
              'split_status': 'unassigned_pending_family_review'}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--linked', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.database, args.linked, args.out), indent=2))
