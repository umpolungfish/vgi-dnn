#!/usr/bin/env python3
"""Make source-bound binary training frames from corpus and evidence records."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import sys

from distinction_format import tokens,OPERATIONS

MAGIC=b'VGIS1\0\0\0'


def pack(frame):
    x,y=frame['tokens'],frame['labels']
    return struct.pack('<4H',frame['kind'],frame['unroll'],frame['steps'],frame['bits'])+\
        struct.pack('<'+'H'*len(x),*x)+struct.pack('<'+'h'*len(y),*y)


def text_windows(row,context,complete=False):
    prefix=[256]+list(row['prompt'].encode('utf8'))+[257]
    target=list(row['target'].encode('utf8'))+[258]
    sequence=prefix+target;block=context//2
    offsets=range(0,len(target),block) if complete else [int(hashlib.sha256(row['id'].encode()).hexdigest()[:8],16)%len(target)]
    for offset in offsets:
        count=min(block,len(target)-offset)
        start=len(prefix)+offset-1-(context-count)
        x=[sequence[i] if 0<=i<len(sequence) else 0 for i in range(start,start+context)]
        labels=[sequence[i+1] if len(prefix)+offset<=i+1<len(prefix)+offset+count else -100
                for i in range(start,start+context)]
        yield {'tokens':x,'labels':labels,'kind':0,'bits':0,'steps':1,'unroll':0,
               'source_id':row['source_id'],'example_id':row['id'],'family':row['family'],
               'task':row['task'],'source':row['source'],'target_byte_offset':offset,
               'target_tokens':count,'target_contains_eos':offset+count==len(target),
               'effective_objective':'local_language_continuation',
               'full_prompt_in_context':start<=0 and start+context>=len(prefix),
               'input_global_span':[start,start+context],
               'source_prompt_token_count':len(prefix),
               'text_content_sha256':hashlib.sha256(json.dumps([row['prompt'],row['target']],ensure_ascii=False).encode()).hexdigest()}


def build(root,views,tasks,output,context=16,limit=None,complete=False,evidence=4):
    if context<8:raise ValueError('context must accommodate typed query fields')
    output.mkdir(parents=True,exist_ok=False);counts=Counter();skips=Counter();seen={};n=0
    stream=output/'train.vgis';manifest=output/'frames.jsonl'
    with stream.open('wb') as data,manifest.open('w',encoding='utf8') as index,\
            (output/'duplicate_aliases.jsonl').open('w',encoding='utf8') as aliases:
        data.write(MAGIC+struct.pack('<I',0))

        def emit(frame):
            nonlocal n
            wire=pack(frame);frame.update({'frame':n,'wire_sha256':hashlib.sha256(wire).hexdigest()})
            data.write(wire);index.write(json.dumps(frame,ensure_ascii=False)+'\n');n+=1;counts[frame['task']]+=1

        for line in (views/'train.jsonl').open():
            row=json.loads(line);task=row['task']
            if row['split']!='train':raise ValueError('nontraining row in training file')
            if limit is not None and counts[task]>=limit:continue
            content=hashlib.sha256(json.dumps([task,row['prompt'],row['target']],ensure_ascii=False).encode()).hexdigest()
            if content in seen:
                skips['exact_task_duplicate']+=1
                aliases.write(json.dumps({'example_id':row['id'],'representative':seen[content],
                    'source_id':row['source_id'],'family':row['family'],'source':row['source']},ensure_ascii=False)+'\n')
                continue
            seen[content]=row['id']
            for frame in text_windows(row,context,complete):
                if limit is not None and counts[task]>=limit:break
                emit(frame)

        # Select real reference records covering independent evidence shapes.
        selected={};policy_rows=[]
        for line in (tasks/'train.jsonl').open():
            row=json.loads(line)
            for t,f in zip(row['support'],row['refute']):selected.setdefault((t,f),row)
            if len(selected)==4:break
        used=set()
        for shape in ((1,1),(0,0),(1,0),(0,1)):
            row=selected[shape]
            if row['id'] not in used:used.add(row['id']);policy_rows.append(row)
        for row in policy_rows[:evidence]:
            x=[0]*(context-8)+tokens(row)
            bits=sum((t+2*f)<<(2*i) for i,(t,f) in enumerate(zip(row['support'],row['refute'])))
            emit({'tokens':x,'labels':[-100]*context,'kind':1,'bits':bits,'steps':1,'unroll':0,
                  'source_id':row['id'],'example_id':row['id'],'family':row['family'],
                  'task':'candidate_evidence','source':{'path':str(tasks/'train.jsonl')},
                  'evidence':row['evidence'],'reference':row})

        sys.path.insert(0,str(root/'VGI/sources/GRAMMAR_UNIFIED_RUN'))
        from carrier_transitions import split,gate,fuse
        for seed,arity,work in [(0,2,'≺'),(3,3,'≻')][:evidence]:
            arms=list(split(seed,arity));arms[0]=gate(work,arms[0]);after=fuse(arms)
            typed=[256,265,266+seed,282+OPERATIONS.index(work),296 if arity==2 else 297,290,0,257]
            emit({'tokens':[0]*(context-8)+typed,'labels':[-100]*context,'kind':2,'bits':after,
                  'steps':1,'unroll':0,'source_id':f'reference:{seed}:{arity}:{work}',
                  'example_id':f'reference:{seed}:{arity}:{work}','task':'carrier_evidence',
                  'source':{'path':'VGI/sources/GRAMMAR_UNIFIED_RUN/carrier_transitions.py'},
                  'reference':{'seed':seed,'arity':arity,'work':work,'after':after},
                  'evidence':'reference_split_work_fuse'})
        data.seek(8);data.write(struct.pack('<I',n))
    bindings={}
    for path in (views/'train.jsonl',views/'report.json',tasks/'train.jsonl',stream,manifest):
        with path.open('rb') as handle:bindings[str(path)]=hashlib.file_digest(handle,'sha256').hexdigest()
    report={'frames':n,'context_length':context,'task_counts':dict(counts),'dispositions':dict(skips),
            'coverage':'all target tokens' if complete else 'deterministic target window per selected record',
            'per_task_limit':limit,'bindings':bindings,'protocol':'VGIS1; fixed token and target arrays',
            'text_refutation_targets':'none; language ranking trains support logits',
            'language_scope':'bounded continuation windows; full-prompt availability is explicit per frame',
            'candidate_carrier_targets':'independent evidence bits',
            'input_split':'train only; source-family assignments retained'}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','views','tasks','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--context',type=int,default=16);p.add_argument('--limit-per-task',type=int)
    p.add_argument('--complete-targets',action='store_true');p.add_argument('--evidence-count',type=int,default=4)
    a=p.parse_args();build(a.root.resolve(),a.views.resolve(),a.tasks.resolve(),a.out.resolve(),a.context,
                          a.limit_per_task,a.complete_targets,a.evidence_count)
