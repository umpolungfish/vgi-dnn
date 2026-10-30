#!/usr/bin/env python3
"""Lift one shared source, oracle-rehearsal and complete-graph Adam update."""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from export_joint_retention import build as build_local
from export_full_learning import compile_model


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root, checkpoint, records, trained_graphs, dataset, library, out,
          graph_weight=1., anchor_weight=4., passes=2, local_weight=.25):
    if not 0 <= graph_weight <= 100 or not 1 <= passes <= 16 or not 0 <= local_weight <= 100:
        raise ValueError('invalid composed joint objective')
    vocabulary = dataset / 'vocabulary.json'
    build_local(root, checkpoint, records, trained_graphs, library, vocabulary, out,
                anchor_weight, 8, 128, True, False, 1)
    runtime = out / 'runtime'
    metadata = json.loads((runtime / 'build.json').read_text())
    rows = [json.loads(line) for line in (out / 'input/records.jsonl').open()]
    source_index = min(range(len(rows)), key=lambda i: len(rows[i]['prefix']) + len(rows[i]['target']))
    graphs = [json.loads(line) for line in (dataset / 'transitions.jsonl').open()]
    graphs = [row for row in graphs if row['split'] == 'train']
    random.Random(113).shuffle(graphs)
    selected = graphs[:16]
    (out / 'composed_training_graphs.json').write_text(json.dumps(selected, indent=2) + '\n')
    ids = {row['name']: row['id'] for row in json.loads(vocabulary.read_text())['tokens']}
    cases = []
    for index, row in enumerate(selected):
        graph = row['graph']; incoming = {}; outgoing = {}
        calls = [f'case {index}:', f'if(vgi_graph_reset({len(graph["nodes"])},{len(graph["edges"])},{graph["readout"]})<0)return -1;']
        for i, name in enumerate(graph['nodes']):
            calls.append(f'if(vgi_graph_node({i},{ids[name]})<0)return -2;')
        for edge, (a,b) in enumerate(graph['edges']):
            calls.append(f'if(vgi_graph_edge({edge},{a},{b},{outgoing.get(a,0)},{incoming.get(b,0)})<0)return -3;')
            outgoing[a] = outgoing.get(a,0) + 1; incoming[b] = incoming.get(b,0) + 1
        # Only topology and seed enter preparation. The linked source gates
        # compute the source return and every local target inside execution.
        calls.append(f'graph_seed={row["seed_bits"]};return 0;')
        cases.append('\n'.join(calls))
    cpp = runtime / 'full_model.cpp'; body = cpp.read_text()
    body = re.sub(r'#define VGI_TAPE_CAPACITY \d+', '#define VGI_TAPE_CAPACITY 16000000', body, count=1)
    body = body[:body.rindex('extern "C" __attribute__((force_align_arg_pointer)) void _start()')]
    declaration = 'extern "C" unsigned int ig_flow_gate(unsigned int,unsigned int,unsigned int,unsigned int,unsigned int);'
    body = body.replace('ig_flow_gate(', 'composed_bound_gate(')
    marker = declaration.replace('ig_flow_gate(', 'composed_bound_gate(')
    assert body.count(marker) == 1
    body = body.replace(marker, declaration + '''
static long composed_oracle_calls,composed_oracle_invocations;
static unsigned int composed_bound_gate(unsigned int a,unsigned int b,unsigned int c,unsigned int d,unsigned int e){
    composed_oracle_calls++;composed_oracle_invocations++;return ig_flow_gate(a,b,c,d,e);}
''', 1)
    insertion = Path(__file__).with_name('composed_graph.hpp').read_text()
    insertion += '\nstatic long joint_composed_configure(long record){switch(record%16){\n' + '\n'.join(cases) + '\n}return -4;}\n'
    marker = 'extern "C" long vgi_joint_corpus_learn(long record){'
    assert body.count(marker) == 1
    body = body.replace(marker, insertion + '\n' + marker, 1)
    marker = 'joint_deferred=0;double norm2=0;'
    assert body.count(marker) == 1
    body = body.replace(marker, f'''if(joint_composed_configure(record)<0){{joint_deferred=0;return -124;}}
    joint_weight={graph_weight:.17g};
    long composed_status=vgi_composed_run({passes},1,{round(local_weight*1000000)},0,1);
    if(composed_status<0){{joint_deferred=0;return composed_status;}}
    joint_deferred=0;double norm2=0;''', 1)
    marker = 'long after=vgi_corpus_learn(record,0,0);'
    assert body.count(marker) == 1
    body = body.replace(marker, f'''if(vgi_composed_run({passes},0,{round(local_weight*1000000)},0,1)<0)return -126;
    long after=vgi_corpus_learn(record,0,0);''', 1)
    body += f'''
extern "C" __attribute__((force_align_arg_pointer)) void _start(){{
    vgi_retained_reset();long result=vgi_joint_corpus_learn({source_index});long status=result<0?2:0;
    if(vgi_composed_save()<0)status=3;
    __asm__ volatile("syscall"::"a"(60L),"D"(status):"rcx","r11","memory");__builtin_unreachable();}}
'''
    cpp.write_text(body)
    metadata['composed_source_joint'] = {'graph_weight': graph_weight, 'passes': passes,
        'local_weight': local_weight, 'source_index': source_index,
        'graph_index': source_index % len(selected), 'graphs': str(out / 'composed_training_graphs.json'),
        'graphs_sha256': sha(out / 'composed_training_graphs.json'),
        'graph_source': str(dataset / 'transitions.jsonl'), 'graph_source_sha256': sha(dataset / 'transitions.jsonl'),
        'source_records_sha256': sha(out / 'input/records.jsonl'),
        'objective': 'complete source mean plus balanced local oracle rehearsal plus complete composed graph; one joint clip and Adam step'}
    metadata.update(source_sha256=sha(cpp), composed_joint_commands=compile_model(root, cpp, runtime, 'O2', library),
                    module_sha256=sha(runtime / 'model.imasm'))
    (runtime / 'build.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'runtime': str(runtime), 'source_index': source_index, 'graph_weight': graph_weight}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root','checkpoint','records','trained-graphs','dataset','library','out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--graph-weight', type=float, default=1.)
    parser.add_argument('--anchor-weight', type=float, default=4.)
    parser.add_argument('--passes', type=int, default=2)
    parser.add_argument('--local-weight', type=float, default=.25)
    a = parser.parse_args()
    build(a.root.resolve(),a.checkpoint.resolve(),a.records.resolve(),a.trained_graphs.resolve(),
          a.dataset.resolve(),a.library.resolve(),a.out.resolve(),a.graph_weight,a.anchor_weight,a.passes,a.local_weight)
