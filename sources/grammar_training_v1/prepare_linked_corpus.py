#!/usr/bin/env python3
"""Preserve catalog and ob3ect records with source-bound links for dataset review.

Uses the existing typed address codec. Does not execute ob3ects or infer verdicts.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from bootstrap import AXES, encode_address, decode_address


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def typed_address(address):
    if not isinstance(address, str):
        raise ValueError('address must be a string')
    address = address.strip()
    if address.startswith('⟨') and address.endswith('⟩'):
        address = address[1:-1]
    index = encode_address(address)
    if decode_address(index) != address:
        raise ValueError('address codec round trip failed')
    return {'address': address, 'crystal_index': index,
            'axis_values': [
                {'axis': symbol, 'axis_name': name, 'glyph': address[i],
                 'value_index': alphabet.index(address[i]), 'cardinality': len(alphabet)}
                for i, (symbol, name, alphabet) in enumerate(AXES)]}


def build(root, output):
    root, output = root.resolve(), output.resolve()
    # An export is a new snapshot, never an overwrite of a previous receipt.
    output.mkdir(parents=True, exist_ok=False)
    manifest, records, links, errors = [], [], [], []
    names, addresses, hashes = defaultdict(list), defaultdict(list), defaultdict(list)

    def load(path, kind):
        raw = path.read_bytes()
        source = {'path': str(path.relative_to(root)), 'sha256': digest(raw),
                  'bytes': len(raw), 'kind': kind}
        manifest.append(source)
        try:
            return json.loads(raw), source
        except (ValueError, UnicodeError) as exc:
            errors.append({'source': source, 'error': str(exc)})
            return None, source

    catalog_path = root / 'imscribing_grammar/IG_catalog.json'
    catalog, source = load(catalog_path, 'catalog')
    if not isinstance(catalog, list):
        raise ValueError('catalog must contain a JSON list')
    for i, entry in enumerate(catalog):
        record_id = f'catalog:{source["sha256"]}:{i}'
        record = {'id': record_id, 'kind': 'catalog_entry',
                  'source': {**source, 'json_pointer': f'/{i}'}, 'payload': entry}
        try:
            record['tuple'] = typed_address(''.join(entry[symbol] for symbol, _, _ in AXES))
        except (KeyError, TypeError, ValueError) as exc:
            errors.append({'record_id': record_id, 'error': str(exc)})
        records.append(record)
        if isinstance(entry, dict) and isinstance(entry.get('name'), str):
            names[entry['name']].append(record_id)
        if 'tuple' in record:
            addresses[record['tuple']['address']].append(record_id)

    # Retain the canonical catalog history alongside its current snapshot.
    for filename in ('IG_catalog.provenance.json', 'IG_catalog.ixcriptions.json'):
        path = catalog_path.with_name(filename)
        if path.exists():
            payload, history_source = load(path, 'catalog_history')
            if payload is not None:
                records.append({'id': f'history:{history_source["sha256"]}',
                                'kind': 'catalog_history', 'source': history_source,
                                'payload': payload})

    object_root = root / 'ob3ect/digital'
    for path in sorted(object_root.rglob('*_ob3ect.json')):
        if path.is_symlink():
            errors.append({'path': str(path.relative_to(root)), 'error': 'symlink not followed'})
            continue
        payload, object_source = load(path, 'ob3ect')
        if payload is None:
            continue
        if not isinstance(payload, dict):
            errors.append({'source': object_source, 'error': 'ob3ect must be a JSON object'})
            continue
        record_id = f'ob3ect:{object_source["path"]}:{object_source["sha256"]}'
        record = {'id': record_id, 'kind': 'ob3ect', 'source': object_source,
                  'payload': payload}
        hashes[object_source['sha256']].append(record_id)
        if payload.get('grounded_tuple') is not None:
            try:
                record['tuple'] = typed_address(payload['grounded_tuple'])
            except (TypeError, ValueError) as exc:
                errors.append({'record_id': record_id, 'field': 'grounded_tuple', 'error': str(exc)})
        # Associated files are inventoried as bytes, without interpreting their claims.
        record['associated_files'] = []
        for sibling in sorted(path.parent.iterdir()):
            if sibling.is_file() and not sibling.is_symlink() and sibling != path:
                record['associated_files'].append(str(sibling.relative_to(root)))
        records.append(record)
        name = payload.get('name')
        for target in names.get(name, []) if isinstance(name, str) else []:
            links.append({'from': record_id, 'to': target, 'relation': 'exact_name_match'})
        if 'tuple' in record:
            for target in addresses.get(record['tuple']['address'], []):
                links.append({'from': record_id, 'to': target, 'relation': 'shared_recorded_tuple'})

    duplicate_groups = [ids for ids in hashes.values() if len(ids) > 1]
    report = {'record_counts': dict(Counter(r['kind'] for r in records)),
              'source_files': len(manifest), 'catalog_unique_addresses': len(addresses),
              'catalog_entries_with_valid_tuple': sum('tuple' in r for r in records if r['kind'] == 'catalog_entry'),
              'ob3ects_with_valid_grounded_tuple': sum('tuple' in r for r in records if r['kind'] == 'ob3ect'),
              'link_counts': dict(Counter(link['relation'] for link in links)),
              'duplicate_ob3ect_hash_groups': len(duplicate_groups), 'errors': len(errors),
              'split_status': 'unassigned_pending_family_review',
              'evidence_status': 'source_records_preserved; no execution performed'}
    for filename, rows in [('manifest.jsonl', manifest), ('records.jsonl', records),
                           ('links.jsonl', links)]:
        with (output / filename).open('w', encoding='utf8') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    for filename, value in [('report.json', report), ('errors.json', errors),
                            ('duplicate_ob3ects.json', duplicate_groups)]:
        (output / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.out), indent=2))
