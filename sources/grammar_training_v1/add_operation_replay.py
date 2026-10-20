#!/usr/bin/env python3
"""Interleave source-bound operation queries with the retained training stream."""
import argparse,hashlib,json,random
from pathlib import Path
from retained_format import stream


def build(batch,records,out,count,cursor,task="graph_local_transition",kind=4,seed=41,balanced=False):
    manifest=json.loads((batch/'manifest.json').read_text());previous=json.loads(cursor.read_text())
    base=[json.loads(l) for l in (batch/'records.jsonl').open()]
    assert stream(base)==(batch/'train.vgir').read_bytes()
    source_hash=hashlib.sha256(records.read_bytes()).hexdigest()
    old=previous.get('auxiliary_sources',{}).get(task,{})
    if old and old['sha256']!=source_hash:raise ValueError('graph operation source changed across continuation')
    candidates=[json.loads(l) for l in records.open()];random.Random(seed).shuffle(candidates)
    if count<0 or count>len(candidates):raise ValueError('graph replay count must fit one complete table epoch')
    groups={}
    if balanced:
        for row in candidates:groups.setdefault(row['reference_bits'],[]).append(row)
        if set(groups)!={0,1,2,3}:raise ValueError('checking replay requires all four evidence shapes')
        if count>4*min(map(len,groups.values())):raise ValueError('balanced replay would repeat an identity inside one batch')
    start=previous['next_task_cursors'].get(task,0);selected=[]
    for ordinal in range(start,start+count):
        row=dict(groups[ordinal%4][(ordinal//4)%len(groups[ordinal%4])]) if balanced else dict(candidates[ordinal%len(candidates)])
        if row['task']!=task or row['training_split']!='train':raise ValueError('auxiliary source task/split mismatch')
        row.update(kind=kind,reference_bits=row.get('reference_bits',row['bits']),bits=0,task_ordinal=ordinal,replay_epoch=ordinal//len(candidates),
                   oracle='declared source operation derives the target during execution')
        selected.append(row)
        if balanced:row.update(evidence_cycle=(ordinal//4)//len(groups[ordinal%4]),schedule='evidence_balanced_round_robin')
    combined=[];position=0
    for index,row in enumerate(base):
        combined.append(row)
        boundary=(index+1)*len(selected)//len(base)
        combined.extend(selected[position:boundary]);position=boundary
    combined.extend(selected[position:]);out.mkdir(parents=True,exist_ok=False)
    payload=stream(combined);(out/'train.vgir').write_bytes(payload)
    (out/'records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in combined))
    manifest['previous_task_cursors'][task]=start;manifest['next_task_cursors'][task]=start+len(selected)
    manifest['task_counts'][task]=len(selected)
    manifest.setdefault('auxiliary_sources',{})[task]={'path':str(records),'sha256':source_hash,'schedule_seed':seed,
                                                  'schedule':'evidence_balanced_round_robin' if balanced else 'shuffled_cyclic'}
    manifest.update(records=len(combined),tokens=sum(len(r['prefix'])+len(r['target']) for r in combined),
                    input_sha256=hashlib.sha256(payload).hexdigest(),interleaving='proportional record positions across the full batch')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'records':len(combined),'operation_queries':len(selected),'task':task}))

