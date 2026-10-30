#!/usr/bin/env python3
"""Prepare whole-record corpus batches with resumable per-task source cursors."""
import argparse,collections,hashlib,json
from pathlib import Path
from retained_format import record,stream

def build(source,out,budget,limit,previous=None,source_index=None):
    if budget<1 or limit<1:raise ValueError("positive token budget and record limit required")
    if previous and 'completed_records' not in json.loads(previous.read_text()):
        raise ValueError('continuation requires a collected runtime cursor, not an unexecuted batch manifest')
    cursors=json.loads(previous.read_text())['next_task_cursors'] if previous else {}
    if source_index:
        from retained_corpus_index import read
        queues,counts,source_hash=read(source,source_index,cursors,limit)
    else:
        queues=collections.defaultdict(list);counts=collections.Counter();digest=hashlib.sha256()
        with source.open('rb') as handle:
            for line in handle:
                digest.update(line);row=json.loads(line);task=row['task'];index=counts[task];counts[task]+=1
                if index>=cursors.get(task,0) and len(queues[task])<limit:
                    queues[task].append((index,row))
        source_hash=digest.hexdigest()
    if previous and json.loads(previous.read_text())['source_sha256']!=source_hash:
        raise ValueError('source changed across batch continuation')
    next_cursors=dict(cursors);next_cursors.update({task:cursors.get(task,0) for task in counts});selected=[];tokens=0;blocked=set()
    # Take each available task before adding further rounds. Never cut a source record.
    for index in range(limit):
        added=False
        for task in sorted(queues):
            if task in blocked or index>=len(queues[task]) or len(selected)>=limit:continue
            ordinal,row=queues[task][index];frame=record(row)
            frame['task_ordinal']=ordinal;frame['training_split']=row['split']
            assert row['split']=='train'
            length=len(frame['prefix'])+len(frame['target'])
            if length>1048576:raise ValueError('whole record exceeds source bank: '+row['id'])
            if index and tokens+length>budget:
                blocked.add(task);continue
            selected.append(frame);tokens+=length;next_cursors[task]=ordinal+1;added=True
        if not added or tokens>=budget or len(selected)>=limit:break
    out.mkdir(parents=True,exist_ok=False)
    payload=stream(selected);(out/'train.vgir').write_bytes(payload)
    (out/'records.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in selected))
    manifest={'source':str(source),'source_index':str(source_index) if source_index else None,'source_sha256':source_hash,'source_task_counts':dict(counts),
              'previous_task_cursors':cursors,'next_task_cursors':next_cursors,'records':len(selected),
              'task_counts':dict(collections.Counter(r['task'] for r in selected)),
              'tokens':tokens,'input_sha256':hashlib.sha256(payload).hexdigest(),
              'conditioning':'complete prompt and target; no source truncation',
              'continuation':'advance cursors only for completed records after runtime receipt',
              'auxiliary_sources':json.loads(previous.read_text()).get('auxiliary_sources',{}) if previous else {}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--token-budget',type=int,default=64000);p.add_argument('--max-records',type=int,default=64)
    p.add_argument('--previous-manifest',type=Path)
    p.add_argument('--index',type=Path,help='exact row-offset index for corpus selection without full rescans')
    a=p.parse_args();build(a.source.resolve(),a.out.resolve(),a.token_budget,a.max_records,a.previous_manifest,a.index.resolve() if a.index else None)
