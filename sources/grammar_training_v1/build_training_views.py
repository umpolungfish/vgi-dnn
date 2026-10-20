#!/usr/bin/env python3
"""Build provenance-linked task views with shared-source family splits."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

from bootstrap import split_for_address


def read_rows(path):
    with path.open(encoding='utf8') as handle:
        for line in handle:
            yield json.loads(line)


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


class Families:
    def __init__(self):
        self.parents = {}

    def find(self, item):
        self.parents.setdefault(item, item)
        if self.parents[item] != item:
            self.parents[item] = self.find(self.parents[item])
        return self.parents[item]

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parents[max(a, b)] = min(a, b)


def build(root, linked, documents, output):
    output.mkdir(parents=True, exist_ok=False)
    records = list(read_rows(linked / 'records.jsonl'))
    files = {r['id']: r for r in read_rows(documents / 'files.jsonl')}
    families, keys = Families(), {}

    def bind(key, record_id):
        families.find(record_id)
        if key in keys:
            families.join(record_id, keys[key])
        else:
            keys[key] = record_id

    for record in records:
        if record['kind'] == 'catalog_history':
            continue  # History remains preserved, pending revision-aware task design.
        rid = record['id']
        bind(('hash', record['source']['sha256']), rid) if record['kind'] == 'ob3ect' else families.find(rid)
        name = record['payload'].get('name')
        if isinstance(name, str) and name:
            bind(('name', name), rid)
        if 'tuple' in record:
            bind(('tuple', record['tuple']['address']), rid)
    for rid, record in files.items():
        source = record['indexed_source']
        bind(('hash', source['sha256']), rid)
        bind(('basename', Path(source['path']).stem.casefold()), rid)
    declaration_sources = {}
    for declaration in read_rows(documents / 'declarations.jsonl'):
        declaration_sources[declaration['id']] = declaration['source_id']
    for link in read_rows(documents / 'links.jsonl'):
        families.join(declaration_sources[link['from']], link['to'])
    del declaration_sources
    assignments = {}
    for rid in families.parents:
        family = families.find(rid)
        assignments[rid] = {'family': family, 'split': split_for_address(family)}

    counters, dispositions = Counter(), Counter()
    handles = {s: (output / f'{s}.jsonl').open('w', encoding='utf8')
               for s in ('train', 'validation', 'test')}

    def emit(task, rid, prompt, target, evidence, source, suffix):
        row = {'id': f'{rid}:{task}:{suffix}', 'task': task, 'source_id': rid,
               **assignments[rid], 'prompt': prompt, 'target': target,
               'evidence': evidence, 'source': source}
        handles[row['split']].write(json.dumps(row, ensure_ascii=False) + '\n')
        counters[f'{row["split"]}:{task}'] += 1

    try:
        for record in records:
            rid, payload = record['id'], record['payload']
            if record['kind'] == 'catalog_entry':
                description = json.dumps({'name': payload.get('name'),
                                          'description': payload.get('description')}, ensure_ascii=False)
                if 'tuple' in record:
                    emit('catalog_to_tuple', rid,
                         'Return the recorded twelve-axis tuple for this catalog entry.\n' + description,
                         record['tuple']['address'], 'catalog_attestation', record['source'], 'tuple')
                    emit('tuple_to_catalog', rid,
                         'Describe the catalog entry named ' + str(payload.get('name')) +
                         ' at tuple ' + record['tuple']['address'] + '.',
                         description, 'catalog_attestation', record['source'], 'meaning')
                else:
                    dispositions['catalog_tuple_pending_review'] += 1
            elif record['kind'] == 'ob3ect':
                phases = payload.get('phases')
                if not isinstance(phases, dict):
                    dispositions['ob3ect_without_phase_dictionary'] += 1
                    continue
                context = {'name': payload.get('name'), 'recorded_phases': {}}
                phase_keys = sorted(phases, key=lambda k: (int(re.search(r'\d+', k)[0])
                                                          if re.search(r'\d+', k) else 999, k))
                for key in phase_keys:
                    emit('ob3ect_recorded_construction', rid,
                         'Continue this recorded ob3ect construction at ' + key + '.\n' +
                         json.dumps(context, ensure_ascii=False),
                         json.dumps(phases[key], ensure_ascii=False),
                         'saved_ob3ect_phase', record['source'], key)
                    context['recorded_phases'][key] = phases[key]
                if 'tuple' in record:
                    emit('ob3ect_to_tuple', rid,
                         'Return the recorded grounded tuple for this ob3ect.\n' +
                         json.dumps(context, ensure_ascii=False), record['tuple']['address'],
                         'saved_ob3ect_grounding', record['source'], 'tuple')

        # Reuse existing extraction for changed sources without modifying the old index.
        sys.path.insert(0, str(root / 'VGI/sources/GRAMMAR_UNIFIED_RUN'))
        from grammar_corpus import decode, chunks
        refreshed = {}
        eligible = set()
        for rid, record in files.items():
            source = record['indexed_source']
            path = Path(source['path'])
            if record['generated_path'] or path.name == 'commit.txt':
                dispositions['generated_or_workflow_file'] += 1
                continue
            if record['source_status'] == 'unavailable' or source['error']:
                dispositions['unavailable_or_failed_extraction'] += 1
                continue
            try:
                current_hash = sha(path)
                if current_hash != record['current_source_sha256']:
                    raise ValueError('source changed since attachment; refresh attachment')
                if record['source_status'] == 'changed':
                    text, extraction = decode(path)
                    refreshed[rid] = (text, {**source, 'sha256': current_hash,
                                             'extraction': extraction, 'refreshed': True})
                    dispositions['refreshed_sources'] += 1
                eligible.add(rid)
            except (OSError, ValueError, RuntimeError) as exc:
                dispositions['source_requires_refresh'] += 1
                with (output / 'refresh_errors.jsonl').open('a', encoding='utf8') as handle:
                    handle.write(json.dumps({'source_id': rid, 'error': str(exc)}) + '\n')

        def emit_text(rid, text, line, ordinal, source):
            if len(text) < 32:
                dispositions['short_text_chunk'] += 1
                return
            split_at = max(16, len(text) // 3)
            task = 'lean_source_continuation' if files[rid]['kind'] == 'lean' else 'document_continuation'
            emit(task, rid, 'Continue this source passage.\n' + text[:split_at],
                 text[split_at:], 'source_text', {**source, 'start_line': line}, str(ordinal))

        for chunk in read_rows(documents / 'chunks.jsonl'):
            rid, payload = chunk['source_id'], chunk['payload']
            if rid in eligible and rid not in refreshed:
                emit_text(rid, payload['content'], payload['start_line'], payload['ordinal'],
                          files[rid]['indexed_source'])
        for rid, (text, source) in refreshed.items():
            for ordinal, line, content in chunks(text):
                emit_text(rid, content, line, ordinal, source)
    finally:
        for handle in handles.values():
            handle.close()
    with (output / 'families.jsonl').open('w', encoding='utf8') as handle:
        for rid, assignment in sorted(assignments.items()):
            handle.write(json.dumps({'source_id': rid, **assignment}, ensure_ascii=False) + '\n')
    sizes = Counter(a['family'] for a in assignments.values())
    report = {'task_counts': dict(counters), 'dispositions': dict(dispositions),
              'families': len(sizes), 'largest_family_sources': max(sizes.values()),
              'input_bindings': {str(p): sha(p) for p in
                                 (linked / 'records.jsonl', documents / 'report.json')},
              'split_grouping': ['recorded tuple', 'exact name', 'identical source bytes',
                                 'same filename stem', 'exact declaration-name link'],
              'remaining_split_review': 'near duplicates, derivations, and graph-family holdouts',
              'index_base': 0, 'source_records_modified': False}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--linked', type=Path, required=True)
    parser.add_argument('--documents', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.root.resolve(), args.linked, args.documents, args.out), indent=2))
