#!/usr/bin/env python3
"""Create honest Grammar-typed neural training records from the native catalog.
No VM executions, proofs or logical verdicts are inferred from tuple positions.
"""
import argparse
import collections
import json
from pathlib import Path
from bootstrap import AXES, encode_address, decode_address, parse_catalog, split_for_address


def build(catalog: Path, output: Path):
    rows, errors = parse_catalog(catalog.read_text(encoding='utf8'))
    output.mkdir(parents=True, exist_ok=True)
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row['address']].append(row)
    stats = collections.Counter()
    paths = {}
    try:
        for split in ('train', 'validation', 'test'):
            paths[split] = (output / f'{split}.jsonl').open('w', encoding='utf8')
        for row in rows:
            addr, name = row['address'], row['name']
            split = split_for_address(addr)
            values = []
            for axis_idx, (symbol, axis_name, choices) in enumerate(AXES):
                values.append({'axis':symbol, 'axis_name':axis_name,
                               'glyph':addr[axis_idx], 'value_index':choices.index(addr[axis_idx]),
                               'cardinality':len(choices)})
            record = {
                'task':'typed_imscription_reconstruction', 'object_name':name,
                'address':addr, 'crystal_index':encode_address(addr),
                'axis_values': values, 'address_multiplicity':len(grouped[addr]),
                'split':split,
                'targets':{'glyphs':list(addr), 'indices':[x['value_index'] for x in values]},
                'evidence':{'source_line':row['source_line'], 'kind':'catalog_attestation',
                            'imasm_trace':None, 'grammar_judgment':None,
                            'kernel_judgment':None,'flow_judgment':None},
            }
            assert decode_address(record['crystal_index']) == addr
            paths[split].write(json.dumps(record, ensure_ascii=False) + '\n')
            stats[split] += 1
    finally:
        for f in paths.values():f.close()
    schema = {'axes':[{'symbol':x[0],'name':x[1],'alphabet':list(x[2]),'radix':len(x[2])} for x in AXES],
              'input':'native_catalog_object',
              'target':'twelve_typed_glyphs_and_crystal_index',
              'note':'No executable IMASM evidence in this export; do not train logical verdicts from this file.'}
    (output/'schema.json').write_text(json.dumps(schema,ensure_ascii=False,indent=2), encoding='utf8')
    (output/'errors.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2), encoding='utf8')
    report={'parsed_objects':len(rows),'invalid_rows':len(errors),
            'unique_addresses':len(grouped),'split_counts':dict(stats),
            'names_with_truncation':sum(row['name_truncated'] for row in rows)}
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    return report

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('catalog',type=Path)
    p.add_argument('--out',type=Path,default=Path('grammar_training'))
    a=p.parse_args()
    print(json.dumps(build(a.catalog,a.out),indent=2))
