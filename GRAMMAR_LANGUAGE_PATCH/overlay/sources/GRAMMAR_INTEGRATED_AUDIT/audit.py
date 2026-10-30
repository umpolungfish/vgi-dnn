#!/usr/bin/env python3
"""Deterministic failure-localization audit for the pretrained coordinate operator.
No imagined VM trace; exact mixed-radix arithmetic is the oracle.
"""
import argparse, collections, json, random, sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parent.parent/'GRAMMAR_INTEGRATED_V1'
if not (ROOT/'integrated_stress.py').is_file():
    raise SystemExit('ERROR: place GRAMMAR_INTEGRATED_AUDIT alongside GRAMMAR_INTEGRATED_V1')
sys.path.insert(0,str(ROOT))
from grammar_ops import RADICES,apply_operation
from integrated_stress import load_ops, load_carrier, carrier_step
from dynamic_bridge import evaluate

def predict_many(model,items,batch_size=256):
    outputs=[]
    with torch.inference_mode():
        for off in range(0,len(items),batch_size):
            batch=items[off:off+batch_size]
            vals=torch.tensor([r['input']+[r['v1'],r['v2']] for r in batch],dtype=torch.long)
            axes=torch.tensor([[r['a'],r['b']] for r in batch],dtype=torch.long)
            modes=torch.tensor([r['mode'] for r in batch],dtype=torch.long)
            logits=model(vals,axes,modes)
            pred=torch.stack([l.argmax(-1) for l in logits],dim=1).tolist()
            outputs.extend(pred)
    return outputs

def audit_one_step(model,samples,seed,batch_size):
    rng=random.Random(seed)
    records=[]
    for i in range(samples):
        x=[rng.randrange(r) for r in RADICES]
        a=rng.randrange(12);b=(a+rng.randrange(1,12))%12
        mode=i%3
        v1=rng.randrange(RADICES[a]);v2=rng.randrange(RADICES[b])
        records.append({'input':x,'a':a,'b':b,'mode':mode,'v1':v1,'v2':v2})
    predictions=predict_many(model,records,batch_size)
    grouped=collections.defaultdict(lambda:[0,0]); wrong_axes=collections.Counter(); failures=[]
    for r,pred in zip(records,predictions):
        truth=apply_operation(r['input'],r['mode'],(r['a'],r['b']),(r['v1'],r['v2']))
        correct=pred==truth
        grouped[str(r['mode'])][0]+=1;grouped[str(r['mode'])][1]+=int(correct)
        if not correct:
            for k,(p,t) in enumerate(zip(pred,truth)):
                if p!=t:wrong_axes[str(k)]+=1
            if len(failures)<24:failures.append({**r,'expected':truth,'predicted':pred,'wrong_axes':[k for k,(p,t) in enumerate(zip(pred,truth)) if p!=t]})
    correct=sum(v[1] for v in grouped.values())
    return {'samples':samples,'exact':correct/samples,'error_count':samples-correct,
            'by_mode':{key:{'samples':v[0],'exact':v[1]/v[0]} for key,v in grouped.items()},
            'wrong_output_axis_counts':dict(wrong_axes),'failures':failures}

def audit_rollout(model,cases,steps,seed):
    rng=random.Random(seed)
    state_pairs=[]
    for case in range(cases):
        x=[rng.randrange(r) for r in RADICES]
        state_pairs.append([x[:],x[:]])
    first=[]; recovered=0; ever=[]; oracle_fails=0; model_fails=0; examples=[]
    for step in range(steps):
        records=[]; truth_next=[]
        for case,(truth,pred) in enumerate(state_pairs):
            a=rng.randrange(12);b=(a+rng.randrange(1,12))%12
            mode=rng.randrange(3);v1=rng.randrange(RADICES[a]);v2=rng.randrange(RADICES[b])
            records.append({'input':pred,'a':a,'b':b,'mode':mode,'v1':v1,'v2':v2})
            truth_next.append(apply_operation(truth,mode,(a,b),(v1,v2)))
        predictions=predict_many(model,records)
        for i,(r,p,new_truth) in enumerate(zip(records,predictions,truth_next)):
            if p!=apply_operation(r['input'],r['mode'],(r['a'],r['b']),(r['v1'],r['v2'])):
                model_fails+=1
            was_wrong=state_pairs[i][0]!=state_pairs[i][1]
            now_wrong=p!=new_truth
            if now_wrong and i not in ever:
                ever.append(i)
                if len(examples)<16:examples.append({'case':i,'step':step+1,'operation':r,'expected':new_truth,'predicted':p})
            if was_wrong and not now_wrong:recovered+=1
            state_pairs[i]=[new_truth,p]
    fully_correct=sum(t==p for t,p in state_pairs)
    return {'cases':cases,'steps':steps,'final_exact':fully_correct/cases,
            'never_diverged':(cases-len(ever))/cases,'ever_diverged':len(ever),
            'local_transition_errors':model_fails,'recovery_events':recovered,
            'first_failure_examples':examples}

def carrier_exhaustive(model):
    errors=[]; n=0
    for seed in range(16):
        for arity in (2,3):
            for mask in range(1<<arity):
                flags=tuple(bool(mask&(1<<i)) for i in range(arity))
                expected=evaluate(seed,arity,flags)['fused_mask']
                actual=carrier_step(model,seed,arity,flags)
                n+=1
                if actual!=expected:errors.append({'seed':seed,'arity':arity,'flags':flags,'expected':expected,'predicted':actual})
    return {'cases':n,'exact':(n-len(errors))/n,'errors':errors[:24]}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ops-checkpoint',default='../IMASM_GRAMMAR_DNN_OPS/operation_checkpoints/best_ops.pt')
    p.add_argument('--carrier-checkpoint',default='../GRAMMAR_CARRIER_LEARNER/carrier_checkpoints/best_carrier.pt')
    p.add_argument('--samples',type=int,default=12000)
    p.add_argument('--cases',type=int,default=256)
    p.add_argument('--steps',type=int,default=256)
    p.add_argument('--seed',type=int,default=20261014)
    p.add_argument('--out',default='audit_report.json')
    a=p.parse_args()
    if min(a.samples,a.cases,a.steps)<1:p.error('samples, cases, steps must be positive')
    torch.set_num_threads(1)
    ops=load_ops(a.ops_checkpoint);carrier=load_carrier(a.carrier_checkpoint)
    report={'scope':'neural coordinate edits and exhaustive carrier truth table',
            'ops_checkpoint':a.ops_checkpoint,'carrier_checkpoint':a.carrier_checkpoint,'seed':a.seed,
            'one_step':audit_one_step(ops,a.samples,a.seed,256),
            'rollout':audit_rollout(ops,a.cases,a.steps,a.seed+1),
            'carrier':carrier_exhaustive(carrier)}
    Path(a.out).write_text(json.dumps(report,indent=2)+'\n')
    for key in ('one_step','rollout','carrier'):
        record=report[key].copy();record.pop('failures',None);record.pop('first_failure_examples',None);record.pop('errors',None)
        print(key.upper(),json.dumps(record),flush=True)
    print('REPORT',a.out,flush=True)
if __name__=='__main__':main()
