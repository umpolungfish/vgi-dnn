#!/usr/bin/env python3
"""Compile and lift the typed candidate operation with attested source binding."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from bootstrap import AXES, encode_address
from export_full_learning import compile_model


def build(root, parent_runtime, records, out):
    rows = [json.loads(s) for s in records.read_text().splitlines()]
    row = next(r for r in rows if r.get('task') == 'catalog_to_tuple' and r.get('training_split') == 'train')
    prompt = bytes(row['prefix'][1:-1]).decode('utf8')
    address = bytes(row['target'][:-1]).decode('utf8'); encode_address(address)
    for key, value in [('prompt_sha256', prompt), ('target_sha256', address)]:
        assert hashlib.sha256(value.encode()).hexdigest() == row[key]
    queries, values, sizes, targets = [], [], [], []
    for axis, (symbol, name, alphabet) in enumerate(AXES):
        text = prompt + '\nTyped coordinates: ' + address[:axis] + '\nAxis ' + symbol + ' (' + name + '): '
        queries.append([256] + list(text.encode()) + [257])
        values.append([list(g.encode()) for g in alphabet])
        sizes.append([len(g.encode()) for g in alphabet])
        targets.append(alphabet.index(address[axis]))
    def array(value):
        return '{' + ','.join(array(v) if isinstance(v, list) else str(v) for v in value) + '}'
    declarations = (f'static unsigned short typed_queries[12][{max(map(len, queries))}]={array(queries)};\n'
                    f'static int typed_query_lengths[12]={array(list(map(len, queries)))};\n'
                    f'static unsigned short typed_values[12][5][4]={array(values)};\n'
                    f'static int typed_value_lengths[12][5]={array(sizes)};\n'
                    f'static int typed_radices[12]={array([len(a[2]) for a in AXES])};\n'
                    f'static int typed_target[12]={array(targets)};\n')
    out.mkdir(parents=True, exist_ok=False)
    body = (parent_runtime / 'full_model.cpp').read_text()
    body = body[:body.rindex('extern "C" __attribute__((force_align_arg_pointer)) void _start()')]
    body = re.sub(r'#define LEARNING_RATE [^\n]+', '#define LEARNING_RATE 0.0001', body, count=1)
    body = re.sub(r'#define VGI_TAPE_CAPACITY \d+', '#define VGI_TAPE_CAPACITY 16000000', body, count=1)
    body += declarations + Path(__file__).with_name('typed_tuple_learning.hpp').read_text()
    body += '''
extern "C" __attribute__((force_align_arg_pointer)) void _start(){
long status=vgi_typed_update()<0?2:0;if(!status&&vgi_write_current()<0)status=3;
__asm__ volatile("syscall"::"a"(60L),"D"(status):"rcx","r11","memory");__builtin_unreachable();}
'''
    source = out / 'full_model.cpp';source.write_text(body)
    metadata = json.loads((parent_runtime / 'build.json').read_text())
    metadata['typed_tuple'] = {'source': row, 'prompt': prompt, 'address': address,
        'source_records_sha256': hashlib.sha256(records.read_bytes()).hexdigest(),
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    metadata['commands'] = compile_model(root, source, out)
    metadata.update(learning_rate=.0001, tape_capacity=16000000,
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    module_sha256=hashlib.sha256((out/'model.imasm').read_bytes()).hexdigest())
    (out / 'build.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','parent-runtime','records','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();build(a.root.resolve(),a.parent_runtime.resolve(),a.records.resolve(),a.out.resolve())
