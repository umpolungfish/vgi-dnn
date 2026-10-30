#!/usr/bin/env python3
"""Hard-example fine-tuning of the pretrained coordinate operator.
No native IMASM claims. Train and evaluation RNG streams are disjoint.
"""
import argparse, json, random, sys
from pathlib import Path
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/'GRAMMAR_INTEGRATED_V1'
if not (ROOT/'grammar_ops.py').exists():
    raise SystemExit('Missing sibling GRAMMAR_INTEGRATED_V1; extract add-on in vox_grammar_induxion')
sys.path.insert(0,str(ROOT))
from grammar_ops import RADICES, apply_operation
from integrated_stress import load_ops

def generate(n,seed):
    rng=random.Random(seed); out=[]
    for i in range(n):
        x=[rng.randrange(r) for r in RADICES]
        a=rng.randrange(12);b=(a+rng.randrange(1,12))%12
        mode=i%3;v1=rng.randrange(RADICES[a]);v2=rng.randrange(RADICES[b])
        out.append((x+[v1,v2],[a,b],mode,apply_operation(x,mode,(a,b),(v1,v2))))
    return out

def tensors(examples):
    return tuple(torch.tensor([r[i] for r in examples],dtype=torch.long) for i in range(4))

def predictions(model, batch):
    vals,axes,mode,_=batch
    return torch.stack([v.argmax(-1) for v in model(vals,axes,mode)],dim=1)

def accuracy(model,examples,batch=256):
    model.eval();correct=0;per=[0,0,0];cnt=[0,0,0];wrong={}
    with torch.inference_mode():
        for off in range(0,len(examples),batch):
            chunk=examples[off:off+batch]; tensors4=tensors(chunk)
            pred=predictions(model,tensors4);target=tensors4[3]
            ok=(pred==target).all(dim=1).tolist()
            for i,good in enumerate(ok):
                mode=chunk[i][2];cnt[mode]+=1;per[mode]+=int(good);correct+=int(good)
                if not good:
                    for ax,(p,t) in enumerate(zip(pred[i].tolist(),target[i].tolist())):
                        if p!=t:wrong[str(ax)]=wrong.get(str(ax),0)+1
    return dict(samples=len(examples),exact=correct/len(examples),errors=len(examples)-correct,
                modes={str(i):{'samples':cnt[i],'exact':per[i]/cnt[i] if cnt[i] else None} for i in range(3)},
                wrong_axes=wrong)

def mine(model,examples,batch=256):
    model.eval();selected=[];hard=[]
    with torch.inference_mode():
        for off in range(0,len(examples),batch):
            chunk=examples[off:off+batch]
            pred=predictions(model,tensors(chunk)).tolist()
            for row,p in zip(chunk,pred):
                if p!=row[3]:hard.append(row)
                elif row[2]!=0:selected.append(row)
    return hard,selected

def run(args):
    torch.set_num_threads(args.threads);torch.manual_seed(args.seed)
    model=load_ops(args.checkpoint)
    holdout=generate(args.holdout,args.seed+1000003)
    baseline=accuracy(model,holdout)
    pool=generate(args.pool,args.seed+27)
    hard,regular=mine(model,pool)
    print('BASELINE',json.dumps(baseline),flush=True)
    print('MINED',json.dumps({'pool':len(pool),'hard':len(hard)}),flush=True)
    rng=random.Random(args.seed+44)
    if not hard:
        print('No failures in mining pool; saving report without fine-tuning.',flush=True)
        report={'baseline':baseline,'after':baseline,'mined_hard':0,'checkpoint_written':False}
    else:
        # Keep original examples in each batch so hard-case training cannot merely memorize failures.
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.001)
        best=baseline['errors'];best_state=None;history=[]
        for epoch in range(1,args.epochs+1):
            model.train()
            for _ in range(args.updates):
                rows=rng.choices(hard,k=args.batch//2)+rng.choices(regular or pool,k=args.batch-args.batch//2)
                rng.shuffle(rows)
                vals,axes,mode,target=tensors(rows)
                optimizer.zero_grad(set_to_none=True)
                logits=model(vals,axes,mode)
                loss=sum(F.cross_entropy(logits[i],target[:,i]) for i in range(12))/12
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            current=accuracy(model,holdout)
            history.append({'epoch':epoch,'errors':current['errors']})
            print('EPOCH',json.dumps(history[-1]),flush=True)
            if current['errors']<best:
                best=current['errors'];best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        # Strict rollback: only accept improvement on untouched holdout; no overwriting original.
        if best_state is not None:
            model.load_state_dict(best_state)
            Path(args.out).parent.mkdir(parents=True,exist_ok=True)
            torch.save({'model':model.state_dict(),'config':model.config,'task':('roundtrip','set_one','set_two'),
                        'provenance':'hard-case coordinate fine-tuning; no IMASM runtime validation'},args.out)
        after=accuracy(model,holdout) if best_state is not None else baseline
        report={'baseline':baseline,'after':after,'mined_hard':len(hard),'checkpoint_written':best_state is not None,
                'candidate_checkpoint':args.out if best_state is not None else None,'history':history,
                'caution':'candidate selected by validation; independent test recorded'}
    final_test=generate(args.holdout,args.seed+2000007)
    report['independent_test']=accuracy(model,final_test)
    Path(args.report).write_text(json.dumps(report,indent=2)+'\n')
    print('FINAL',json.dumps({k:v for k,v in report.items() if k!='history'}),flush=True)
    print('REPORT',args.report,flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='../IMASM_GRAMMAR_DNN_OPS/operation_checkpoints/best_ops.pt')
    p.add_argument('--out',default='coordinate_checkpoints/candidate_ops.pt')
    p.add_argument('--report',default='repair_report.json')
    p.add_argument('--pool',type=int,default=30000);p.add_argument('--holdout',type=int,default=12000)
    p.add_argument('--epochs',type=int,default=8);p.add_argument('--updates',type=int,default=80)
    p.add_argument('--batch',type=int,default=128);p.add_argument('--lr',type=float,default=.0001)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--seed',type=int,default=10429)
    a=p.parse_args()
    if min(a.pool,a.holdout,a.epochs,a.updates,a.batch,a.threads)<1:p.error('all count arguments must be positive')
    run(a)
if __name__=='__main__':main()
