#!/usr/bin/env python3
"""Select complete source records across corpus views for retained learning."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from retained_format import record,stream
from distinction_format import tokens


def build(views,tasks,output):
    output.mkdir(parents=True,exist_ok=False);records=[];seen=set()
    for line in (views/'train.jsonl').open():
        row=json.loads(line)
        if row['task'] not in seen:
            records.append(record(row));seen.add(row['task'])
        if len(seen)==6:break
    # Preserve positive and negative evidence in the same target.
    candidate=next(row for row in map(json.loads,(tasks/'train.jsonl').open())
                   if any(t and f for t,f in zip(row['support'],row['refute'])))
    bits=sum((t+2*f)<<(2*i) for i,(t,f) in enumerate(zip(candidate['support'],candidate['refute'])))
    records.append({'prefix':tokens(candidate),'target':[],'kind':1,'bits':bits,'steps':1,'checkpoint_chunks':4,
                    'task':'candidate_evidence','source_id':candidate['id'],'example_id':candidate['id'],
                    'reference':candidate,'source':{'path':str(tasks/'train.jsonl')}})
    wire=stream(records);(output/'train.vgir').write_bytes(wire)
    with (output/'records.jsonl').open('w',encoding='utf8') as handle:
        for row in records:handle.write(json.dumps(row,ensure_ascii=False)+'\n')
    report={'records':len(records),'task_counts':dict(Counter(r['task'] for r in records)),
            'prefix_tokens':sum(len(r['prefix']) for r in records),'target_tokens':sum(len(r['target']) for r in records),
            'maximum_record_tokens':max(len(r['prefix'])+len(r['target']) for r in records),
            'source_conditioning':'whole prompt; no prompt or target truncation',
            'gradient_scope':'full-source gradients via four-chunk recomputation checkpoints',
            'stream_sha256':hashlib.sha256(wire).hexdigest()}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('views','tasks','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();build(a.views.resolve(),a.tasks.resolve(),a.out.resolve())
