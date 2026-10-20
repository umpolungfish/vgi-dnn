#!/usr/bin/env python3
"""Stratified complete-source validation, with tuple and text units separate."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
import torch
from optimizer_state import load_model
from retained_conditioning import evaluate
from retained_format import record
from typed_tuple_learning import objective, infer


def measure(root, corpus, checkpoints, out, per_task=2, seed=83):
    if per_task < 1:
        raise ValueError('positive sample size required')
    rng=random.Random(seed);pools=defaultdict(list);counts=defaultdict(int)
    # Reservoir sampling ranges over every validation row. Source records remain
    # whole, with their original identities, provenance and split boundaries.
    for line in corpus.open():
        row=json.loads(line);assert row['split']=='validation'
        task=row['task'];counts[task]+=1
        if len(pools[task])<per_task:pools[task].append(row)
        else:
            slot=rng.randrange(counts[task])
            if slot<per_task:pools[task][slot]=row
    torch.set_num_threads(1);sys.path.insert(0,str(root/'bob3ect/reentry_dnn'))
    from dtransformer_torch import DialetheicTransformerTorch
    results=[]
    with torch.no_grad():
        for checkpoint in checkpoints:
            model=DialetheicTransformerTorch(**json.loads((checkpoint/'config.json').read_text())).double()
            model.load_state_dict(load_model(checkpoint/'model.safetensors'));model.eval()
            tasks={}
            for task,sources in sorted(pools.items()):
                rows=[]
                for source in sources:
                    if task in ('catalog_to_tuple','ob3ect_to_tuple'):
                        loss,_=objective(model,source['prompt'],source['target'])
                        proposal=infer(model,source['prompt'])
                        rows.append({'source':source,'axis_loss':float(loss),'prediction':proposal,
                            'correct_coordinates':sum(a==b for a,b in zip(source['target'],proposal['address'])),
                            'exact':source['target']==proposal['address']})
                    else:
                        frame=record(source);value=evaluate(model,frame['prefix'],frame['target'],context=16)
                        rows.append({'source':source,'text_loss':value['loss'],'labels':value['labels'],
                                     'source_tokens_processed':value['source_tokens_processed']})
                if task in ('catalog_to_tuple','ob3ect_to_tuple'):
                    summary={'axis_loss':sum(r['axis_loss'] for r in rows)/len(rows),
                        'correct_coordinates':sum(r['correct_coordinates'] for r in rows),
                        'coordinates':12*len(rows),'exact':sum(r['exact'] for r in rows)}
                else:
                    summary={'mean_record_text_loss':sum(r['text_loss'] for r in rows)/len(rows),
                        'label_mean_text_loss':sum(r['text_loss']*r['labels'] for r in rows)/sum(r['labels'] for r in rows)}
                tasks[task]={'summary':summary,'rows':rows}
                print(json.dumps({'checkpoint':str(checkpoint),'task':task,**summary}),flush=True)
            text=[v['summary']['mean_record_text_loss'] for v in tasks.values() if 'mean_record_text_loss' in v['summary']]
            results.append({'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256((checkpoint/'model.safetensors').read_bytes()).hexdigest(),
                            'tasks':tasks,'task_balanced_text_loss':sum(text)/len(text)})
    report={'scope':'fixed stratified complete validation records; tuple value accuracy separate from text loss; no updates',
        'corpus':str(corpus),'corpus_sha256':hashlib.sha256(corpus.read_bytes()).hexdigest(),
        'seed':seed,'per_task':per_task,'population_by_task':dict(counts),'results':results}
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','corpus','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--checkpoints',type=Path,nargs='+',required=True)
    p.add_argument('--per-task',type=int,default=2);p.add_argument('--seed',type=int,default=83)
    a=p.parse_args();measure(a.root.resolve(),a.corpus.resolve(),[c.resolve() for c in a.checkpoints],a.out.resolve(),a.per_task,a.seed)
