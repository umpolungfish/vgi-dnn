#!/usr/bin/env python3
"""Bake complete language records and source-bound graph gradients into joint updates."""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from export_full_learning import export, compile_model
from retained_format import stream


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root, checkpoint, records, trained_graphs, library, vocabulary, out, weight, anchors, anchor_pool=None, balanced_evidence=False, all_records=False, snapshot_records=1, learning_rate=.001):
    if not 0 <= weight <= 100 or anchors < 1:
        raise ValueError('nonnegative graph weight up to 100 and positive anchor count required')
    rows = [json.loads(line) for line in records.open()]
    primary = rows if all_records else [row for row in rows if row['kind'] == 0]
    if not any(row['kind'] == 0 for row in primary) or snapshot_records < 1:
        raise ValueError('no complete language source records')
    if any(row['kind'] == 0 and row['steps'] != 1 for row in primary):
        raise ValueError('joint source records require one declared optimizer step')
    graphs = [json.loads(line) for line in trained_graphs.open()]
    if any(row['split'] != 'train' for row in graphs):
        raise ValueError('graph loss may use only training graphs')
    tokens = {row['name']: row['id'] for row in json.loads(vocabulary.read_text())['tokens']}
    groups = defaultdict(list)
    for row in graphs:
        graph = row['graph']; fans = {}; slots = {}
        for a, _ in graph['edges']:
            fans[a] = fans.get(a, 0) + 1
        for edge, (a, b) in enumerate(graph['edges']):
            slot = slots.get(a, 0); slots[a] = slot + 1
            query = [256, 71, tokens[graph['nodes'][a]], 266 + row['node_inputs'][a],
                     266 + row['seed_bits'], fans[a], slot, 257]
            groups[row['edge_values'][edge]].append(
                {'graph_id': row['id'], 'family': graph['family'], 'split': row['split'],
                 'edge': edge, 'from': a, 'to': b, 'query': query,
                 'source_bits': row['edge_values'][edge]})
    rng = random.Random(97)
    for group in groups.values():
        rng.shuffle(group)
    anchor_pool = anchor_pool or anchors
    if anchor_pool < anchors:
        raise ValueError('anchor pool must contain one full joint update')
    order = sorted(groups)
    if balanced_evidence:
        if set(groups) != set(range(16)) or anchors % 2 or anchor_pool % 16:
            raise ValueError('balanced evidence needs every source mask, an even update size, and a pool divisible by sixteen')
        order = [mask for value in range(8) for mask in (value, value ^ 15)]
    selected = []; used = set()
    while len(selected) < anchor_pool:
        added = False
        for bits in order:
            found = False
            while groups[bits]:
                row = groups[bits].pop()
                key = tuple(row['query'])
                if key not in used:
                    selected.append(row); used.add(key); added = found = True; break
            if balanced_evidence and not found:
                raise ValueError('balanced evidence pool exhausted a distinct source mask')
            if len(selected) == anchor_pool:
                break
        if not added:
            raise ValueError('insufficient distinct source-bound graph queries')
    if balanced_evidence:
        for record in range(len(primary)):
            masks = [selected[(record * anchors + i) % anchor_pool]['source_bits'] for i in range(anchors)]
            assert all(sum(bool(mask & (1 << bit)) for mask in masks) == anchors // 2 for bit in range(4))
    out.mkdir(parents=True, exist_ok=False)
    source = out / 'input'; source.mkdir()
    (source / 'records.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in primary))
    (source / 'train.vgir').write_bytes(stream(primary))
    manifest = json.loads((records.parent / 'manifest.json').read_text())
    counts = {task: sum(row['task'] == task for row in primary) for task in {row['task'] for row in primary}}
    manifest.update(records=len(primary), tokens=sum(len(row['prefix']) + len(row['target']) for row in primary),
                    task_counts=counts, input_sha256=sha(source / 'train.vgir'))
    (source / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (out / 'graph_anchors.json').write_text(json.dumps(selected, indent=2) + '\n')
    runtime = out / 'runtime'
    export(root, checkpoint, runtime, retained=True, retained_settle=True, context_length=16,
           optimization='O2', snapshot_records=snapshot_records, corpus_records=source / 'records.jsonl',
           corpus_entry=True, graph_library=library, graph_vocabulary=vocabulary, optimizer='adam',learning_rate=learning_rate)
    cpp = runtime / 'full_model.cpp'; body = cpp.read_text()
    marker = 'static void optimizer_apply(const double*gradient,double scale){'
    if body.count(marker) != 1:
        raise ValueError('shared optimizer entry changed')
    body = body.replace(marker, 'static void optimizer_apply_base(const double*gradient,double scale){', 1)
    wrapper = '''
static int joint_deferred;
static double joint_weight,joint_gradient[PARAMETERS];
static void optimizer_apply(const double*gradient,double scale){
    if(joint_deferred){for(int i=0;i<PARAMETERS;i++)joint_gradient[i]+=joint_weight*gradient[i];return;}
    optimizer_apply_base(gradient,scale);
}
'''
    observe = 'extern "C" double vgi_optimizer_observe'
    if body.count(observe) != 1:
        raise ValueError('shared optimizer observation entry changed')
    body = body.replace(observe, wrapper + '\n' + observe, 1)
    body = body[:body.rindex('extern "C" __attribute__((force_align_arg_pointer)) void _start()')]
    body += '\nstatic const unsigned short joint_queries[][8]={' + ','.join(
        '{' + ','.join(map(str, row['query'])) + '}' for row in selected) + '};\n'
    body += f'#define JOINT_COUNT {anchors}\n#define JOINT_POOL_COUNT {anchor_pool}\n#define JOINT_WEIGHT {format(weight,".17g")}\n#define JOINT_SNAPSHOT_INTERVAL {snapshot_records}\n'
    body += '''
static void joint_log(long record,long query,long target,long prediction,long loss){
    char line[180];int length=0;const char*prefix="vgi-joint ";while(*prefix)line[length++]=*prefix++;
    long fields[5]={record,query,target,prediction,loss};
    for(int k=0;k<5;k++){
        long value=fields[k];if(value<0){line[length++]='-';value=-value;}
        char digits[24];int n=0;do{digits[n++]=(char)('0'+value%10);value/=10;}while(value);
        while(n)line[length++]=digits[--n];line[length++]=k==4?'\\n':' ';
    }
    long ignored;__asm__ volatile("syscall":"=a"(ignored):"a"(1L),"D"(2L),"S"(line),"d"((unsigned long)length):"rcx","r11","memory");
}
extern "C" long vgi_joint_corpus_learn(long record){
    if(record<0||(unsigned long)record>=corpus_count)return -120;
    if(corpus_records[record].kind!=0)return vgi_corpus_learn(record,corpus_records[record].steps,0);
    const auto&r=corpus_records[record];unsigned long previous=retained_records;
    for(int i=0;i<PARAMETERS;i++)joint_gradient[i]=0;
    joint_deferred=1;joint_weight=1;
    long primary=vgi_corpus_learn(record,1,0);
    if(primary<0){joint_deferred=0;return primary;}
    joint_weight=((double)JOINT_WEIGHT)/JOINT_COUNT;
    for(int i=0;i<JOINT_COUNT;i++){
        const unsigned short*q=joint_queries[((unsigned long)record*JOINT_COUNT+i)%JOINT_POOL_COUNT];
        if(retained_graph_local_validate(q)<0){joint_deferred=0;return -121;}
        for(int k=0;k<8;k++)retained_bank[k]=q[k];
        long target=ig_flow_gate(q[2],q[3]-266,q[4]-266,q[6],q[5]);
        if(target<0||target>15){joint_deferred=0;return -122;}
        long loss=vgi_retained_update(8,0,2,target,1,1);
        if(loss<0){joint_deferred=0;return loss;}
        long prediction=0;for(int bit=0;bit<4;bit++)prediction|=((bit%2?retained_first_f[259+bit/2]:retained_first_t[259+bit/2])>0)<<bit;
        joint_log(record,((unsigned long)record*JOINT_COUNT+i)%JOINT_POOL_COUNT,target,prediction,loss);
    }
    joint_deferred=0;double norm2=0;
    for(int i=0;i<PARAMETERS;i++){
        if(!finite_number(joint_gradient[i]))return -123;
        norm2+=joint_gradient[i]*joint_gradient[i];initial_gradient[i]=joint_gradient[i];
    }
    if(!finite_number(norm2))return -123;
    double norm=raw_sq(norm2),scale=(STREAM_CLIP>0&&norm>STREAM_CLIP)?STREAM_CLIP/norm:1;
    optimizer_apply_base(joint_gradient,scale);
    long after=vgi_corpus_learn(record,0,0);if(after<0)return after;
    retained_records=previous+1;retained_committed=retained_requested=1;
    return primary;
}
extern "C" __attribute__((force_align_arg_pointer)) void _start(){
    vgi_retained_reset();long status=0;
    for(unsigned long i=0;i<corpus_count;i++){
        long loss=vgi_joint_corpus_learn(i);stream_steps_completed=retained_committed;
        stream_log(i,loss,corpus_records[i].kind);if(loss<0){status=2;break;}
        if(retained_records%JOINT_SNAPSHOT_INTERVAL==0)
            if(vgi_write_current()<0||retained_write_state()<0){status=3;break;}
    }
    if(!status)vgi_retained_settle(1);
    if(vgi_write_current()<0||retained_write_state()<0)status=3;
    __asm__ volatile("syscall"::"a"(60L),"D"(status):"rcx","r11","memory");__builtin_unreachable();}
'''
    cpp.write_text(body)
    report = json.loads((runtime / 'build.json').read_text())
    report.update(joint_retention={'weight': weight, 'anchors': anchors, 'anchor_pool': anchor_pool,
                                  'balanced_evidence_per_update': balanced_evidence,
                                  'all_source_records': all_records,
                                  'execution_trace': 'vgi-joint record query source_bits proposal_bits loss_millionths',
                                  'anchor_source': str(trained_graphs), 'anchor_source_sha256': sha(trained_graphs),
                                  'anchor_records': str(out / 'graph_anchors.json'),
                                  'anchor_records_sha256': sha(out / 'graph_anchors.json'),
                                  'objective': 'complete source mean loss plus weight times mean source-graph bit loss; one global clip and Adam step'},
                  joint_compile_commands=compile_model(root, cpp, runtime, 'O2', library),
                  source_sha256=sha(cpp), module_sha256=sha(runtime / 'model.imasm'))
    (runtime / 'build.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'runtime': str(runtime), 'records': len(primary), 'anchors': anchors, 'anchor_pool': anchor_pool, 'weight': weight}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'checkpoint', 'records', 'trained-graphs', 'library', 'vocabulary', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--weight', type=float, default=1.); p.add_argument('--anchors', type=int, default=8)
    p.add_argument('--anchor-pool', type=int, help='rotate each record through this many distinct source-bound graph queries')
    p.add_argument('--balanced-evidence', action='store_true', help='pair complementary source masks so every joint step sees equal present and absent evidence in each bit')
    p.add_argument('--all-records', action='store_true', help='also execute every distinction, local graph, and checking record through its existing source oracle')
    p.add_argument('--snapshot-records', type=int, default=1)
    p.add_argument('--learning-rate', type=float, default=.001)
    a = p.parse_args()
    build(a.root.resolve(), a.checkpoint.resolve(), a.records.resolve(), a.trained_graphs.resolve(),
          a.library.resolve(), a.vocabulary.resolve(), a.out.resolve(), a.weight, a.anchors, a.anchor_pool, a.balanced_evidence, a.all_records, a.snapshot_records,a.learning_rate)
