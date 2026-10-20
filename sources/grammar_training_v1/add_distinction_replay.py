#!/usr/bin/env python3
"""Interleave training-family graph evidence with a complete-source corpus batch."""
import argparse,hashlib,json,random
from pathlib import Path
from distinction_format import tokens
from retained_format import stream

def build(batch,tasks,out,count,cursor=None,live_oracle=False):
    manifest=json.loads((batch/'manifest.json').read_text())
    corpus=[json.loads(l) for l in (batch/'records.jsonl').open()]
    assert (batch/'train.vgir').read_bytes()==stream(corpus)
    source=tasks/'train.jsonl';digest=hashlib.sha256(source.read_bytes()).hexdigest()
    previous=json.loads(cursor.read_text()) if cursor else {}
    old_hash=previous.get('auxiliary_sources',{}).get('candidate_evidence',{}).get('sha256')
    if old_hash and old_hash!=digest:raise ValueError('distinction source changed across continuation')
    rows=[json.loads(l) for l in source.open()];random.Random(29).shuffle(rows)
    start=previous.get('next_task_cursors',{}).get('candidate_evidence',0)
    if not rows or count<0 or count>len(rows):raise ValueError('replay count must fit one complete query epoch')
    selected=[]
    for ordinal in range(start,start+count):
        row=rows[ordinal%len(rows)]
        assert row['split']=='train'
        bits=sum((int(row['support'][k])<<(2*k))|(int(row['refute'][k])<<(2*k+1)) for k in range(4))
        selected.append({'prefix':tokens(row),'target':[],'kind':1,'bits':bits,'steps':1,'checkpoint_chunks':4,
            'task':'candidate_evidence','replay_epoch':ordinal//len(rows),'task_ordinal':ordinal,'training_split':'train','family':row['family'],
            'source_id':row['id'],'example_id':row['id'],'source':{'path':str(source),'sha256':digest},
            'reference':row,'conditioning':'label-free query tokens; independently supervised support/refutation evidence'})
    if live_oracle:
        for frame in selected:
            frame.update(kind=3,bits=0,oracle='targets computed during execution from the query and graph operations')
    mixed=[];position=0
    for row in corpus:
        mixed.append(row);chunk=(len(selected)+len(corpus)-1)//len(corpus)
        mixed.extend(selected[position:position+chunk]);position+=chunk
    mixed.extend(selected[position:]);out.mkdir(parents=True,exist_ok=False)
    payload=stream(mixed);(out/'train.vgir').write_bytes(payload)
    (out/'records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in mixed))
    manifest['next_task_cursors']['candidate_evidence']=start+len(selected)
    manifest['previous_task_cursors']['candidate_evidence']=start
    manifest.update(records=len(mixed),tokens=sum(len(r['prefix'])+len(r['target']) for r in mixed),
        input_sha256=hashlib.sha256(payload).hexdigest(),auxiliary_sources={**manifest.get('auxiliary_sources',{}),'candidate_evidence':{'path':str(source),'sha256':digest,'schedule_seed':29}})
    manifest['task_counts']['candidate_evidence']=len(selected)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'records':len(mixed),'candidate_training_records':len(selected),'tokens':manifest['tokens']}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('batch','tasks','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--count',type=int,default=512);p.add_argument('--cursor',type=Path)
    p.add_argument('--live-oracle',action='store_true',help='derive targets and decide correction inside the executing module')
    a=p.parse_args();build(a.batch.resolve(),a.tasks.resolve(),a.out.resolve(),a.count,a.cursor,a.live_oracle)
