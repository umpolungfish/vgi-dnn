#!/usr/bin/env python3
"""Grammar-native catalog compiler: strict 12-axis Shavian validation.

This is a data compiler, not an IMASM execution engine. It deliberately makes
no inference of operations, closure, flow, or proofs from coordinate distance.
"""
import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

AXES = [
    ('⊢', 'dimensionality', '𐑛𐑨𐑼𐑦'),
    ('⊣', 'topology', '𐑡𐑰𐑥𐑶𐑸'),
    ('≻', 'coupling', '𐑩𐑑𐑽𐑾'),
    ('≺', 'parity', '𐑗𐑿𐑬𐑯𐑹'),
    ('⋈', 'fidelity', '𐑱𐑞𐑐'),
    ('⊤', 'kinetics', '𐑘𐑤𐑧𐑪𐑺'),
    ('∈', 'cardinality', '𐑚𐑔𐑲'),
    ('∋', 'composition', '𐑝𐑜𐑠𐑵'),
    ('⊙', 'criticality', '𐑢⊙𐑮𐑻𐑣'),
    ('⊥', 'chirality', '𐑓𐑒𐑖𐑫'),
    ('⊞', 'stoichiometry', '𐑙𐑕𐑳'),
    ('⊡', 'winding', '𐑷𐑴𐑭𐑟'),
]
assert [len(x[2]) for x in AXES] == [4,5,4,5,3,5,3,4,5,4,3,4]
assert len(set(''.join(axis[2] for axis in AXES))) == 49
assert __import__('math').prod(len(axis[2]) for axis in AXES) == 17_280_000

# Induxion §5: canonical mixed-radix address codec.
RADICES = tuple(len(symbols) for _, _, symbols in AXES)
STRIDES = (4_320_000, 864_000, 216_000, 43_200, 14_400, 2_880,
           960, 240, 48, 12, 4, 1)
# Induxion §3-4: provenance of each invariant's stage and producing operation.
INDUXION_PROVENANCE = (
    ("Albedo", "I4"), ("Rubedo", "L3"), ("Nigredo", "L1+L2"),
    ("Citrinitas/Rubedo", "A1+L5"), ("Citrinitas", "A3"),
    ("Citrinitas", "A2"), ("Albedo", "I1"), ("Citrinitas", "A4"),
    ("Rubedo", "L4+L6"), ("Albedo", "I2"), ("Citrinitas", "A1"),
    ("Albedo", "I3"),
)

def encode_address(shavian_word):
    """Map one complete twelve-coordinate Shavian type to [0, 17279999]."""
    if len(shavian_word) != len(AXES):
        raise ValueError("address must have exactly 12 glyphs")
    digits = []
    for glyph, (_, _, alphabet) in zip(shavian_word, AXES):
        if glyph not in alphabet:
            raise ValueError(f"glyph {glyph!r} invalid for its axis")
        digits.append(alphabet.index(glyph))
    return sum(digit * stride for digit, stride in zip(digits, STRIDES))

def decode_address(index):
    """Recover a Shavian tuple from its Induxion mixed-radix address."""
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < 17_280_000:
        raise ValueError("address index outside crystal")
    result = []
    for (_, _, alphabet), stride in zip(AXES, STRIDES):
        digit, index = divmod(index, stride)
        result.append(alphabet[digit])
    return "".join(result)

assert RADICES == (4,5,4,5,3,5,3,4,5,4,3,4)

ROW = re.compile(r'([^│\n]+)\s*│\s*⟨([^⟩]+)⟩\s*│\s*(\S+)')

def parse_catalog(text):
    rows, errors = [], []
    for line_no, line in enumerate(text.splitlines(), 1):
        if '⟨' not in line:
            continue
        m = ROW.search(line)
        if not m:
            errors.append({'line': line_no, 'error': 'malformed row'})
            continue
        name, address, display_first = (x.strip() for x in m.groups())
        name = name.strip('│ ').strip()
        issues=[]
        if len(address) != 12:
            issues.append('address length != 12')
        else:
            issues.extend(f'axis {i+1} ({axis[1]}) rejects {address[i]!r}' for i,axis in enumerate(AXES) if address[i] not in axis[2])
            if display_first != address[0]: issues.append('first glyph mismatch')
        if issues:
            errors.append({'line':line_no,'name':name,'errors':issues})
            continue
        rows.append({'name':name,'address':address,'source_line':line_no,
                     'name_truncated':'…' in name,
                     'ordinals':[axis[2].index(address[i])+1 for i,axis in enumerate(AXES)],
                     'crystal_index':encode_address(address)})
    return rows, errors

def split_for_address(address):
    # Keep all object names sharing one address in same split.
    h=int(hashlib.sha256(address.encode('utf-8')).hexdigest()[:8],16)%10
    return 'test' if h==0 else 'validation' if h==1 else 'train'

def compile_catalog(source, output):
    rows, errors=parse_catalog(source.read_text(encoding='utf8'))
    output.mkdir(parents=True,exist_ok=True)
    by_addr=collections.defaultdict(list)
    for row in rows: by_addr[row['address']].append(row)
    names=collections.defaultdict(set)
    for row in rows: names[row['name']].add(row['address'])
    pairs=[]
    # Efficient Hamming distance=1 enumeration, with explicit label: proximity only.
    seen=set()
    for addr in by_addr:
        for axis_idx, (_, axis_name, symbols) in enumerate(AXES):
            for other_symbol in symbols:
                if other_symbol == addr[axis_idx]: continue
                neighbor=addr[:axis_idx]+other_symbol+addr[axis_idx+1:]
                if neighbor not in by_addr: continue
                key=tuple(sorted((addr,neighbor)))
                if key in seen: continue
                seen.add(key)
                pairs.append({'kind':'coordinate_neighbor_not_proven_composition',
                              'address_a':addr,'address_b':neighbor,'changed_axis':axis_name,
                              'axis_index':axis_idx,'names_a':[r['name'] for r in by_addr[addr]],
                              'names_b':[r['name'] for r in by_addr[neighbor]]})
    with (output/'objects.jsonl').open('w',encoding='utf8') as f:
        for row in rows:
            record={'kind':'catalog_object','source':'user_catalog_excerpt',**row,
                    'split':split_for_address(row['address']),
                    'execution_trace':None,'kernel_verdict':None,'flow_verdict':None}
            f.write(json.dumps(record,ensure_ascii=False)+'\n')
    with (output/'neighbors.jsonl').open('w',encoding='utf8') as f:
        for item in pairs:f.write(json.dumps(item,ensure_ascii=False)+'\n')
    with (output/'collisions.jsonl').open('w',encoding='utf8') as f:
        for addr,group in sorted(by_addr.items(),key=lambda kv:-len(kv[1])):
            if len(group)>1:
                f.write(json.dumps({'address':addr,'count':len(group),'names':[r['name'] for r in group]},ensure_ascii=False)+'\n')
    (output/'schema.json').write_text(json.dumps({
        'source': 'INDUXION(1).pdf §§3-5 + SNS_PRIME.md',
        'axis_order': [axis[0] for axis in AXES],
        'radices': RADICES, 'strides': STRIDES,
        'axes': [{'glyph': axis[0], 'name': axis[1], 'values': list(axis[2]),
                  'stage': stage, 'induxion_operations': op}
                 for axis, (stage, op) in zip(AXES, INDUXION_PROVENANCE)]
    },indent=2,ensure_ascii=False),encoding='utf8')
    (output/'errors.json').write_text(json.dumps(errors,indent=2,ensure_ascii=False),encoding='utf8')
    stats={'parsed_objects':len(rows),'unique_addresses':len(by_addr),
           'repeated_address_extra_entries':len(rows)-len(by_addr),
           'collision_groups':sum(len(v)>1 for v in by_addr.values()),
           'coordinate_neighbor_edges':len(pairs),'invalid_rows':len(errors),
           'potential_name_truncations':sum(r['name_truncated'] for r in rows),
           'address_capacity':17_280_000,'occupied_fraction':len(by_addr)/17_280_000,
           'splits':dict(collections.Counter(split_for_address(r['address']) for r in rows)),
           'top_collision_groups':[{'address':a,'count':len(v),'names':[r['name'] for r in v[:8]]} for a,v in sorted(by_addr.items(),key=lambda kv:-len(kv[1]))[:10]]}
    (output/'report.json').write_text(json.dumps(stats,indent=2,ensure_ascii=False),encoding='utf8')
    return stats

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('catalog',type=Path)
    p.add_argument('--out',type=Path,default=Path('compiled'))
    a=p.parse_args()
    print(json.dumps(compile_catalog(a.catalog,a.out),ensure_ascii=False,indent=2))
