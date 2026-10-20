#!/usr/bin/env python3
"""Evaluate held-out crystal edits as multi-step free rollout, against an exact oracle.

Requires grammar_ops.py, grammar_dnn.py, a trained operation_checkpoints/best_ops.pt,
and grammar_training/test.jsonl from IMASM_GRAMMAR_DNN_OPS.
No claim of IMASM VM / SIXTEEN_3 execution is made.
"""
import argparse, json, random
from pathlib import Path
import torch
from grammar_ops import CrystalReentry, RADICES, apply_operation, index_from_digits, digits_from_index
from grammar_dnn import read_records


def make_schedule(seed, n, length, rng, kind):
    schedule=[]
    for step in range(length):
        axes=[];assigned=[]
        for _ in range(n):
            axis=rng.randrange(12)
            # 'all' includes edits that leave the digit unchanged, testing idempotence.
            v=rng.randrange(RADICES[axis])
            axes.append(axis);assigned.append(v)
        schedule.append((axes,assigned))
    return schedule


def forward_step(model, state, ax, value, device):
    n=len(state)
    vals=torch.tensor([s+[v,0] for s,v in zip(state,value)],dtype=torch.long,device=device)
    axes=torch.tensor([[a,0] for a in ax],dtype=torch.long,device=device)
    mode=torch.ones(n,dtype=torch.long,device=device)
    with torch.no_grad():
        logits=model(vals,axes,mode)
        return torch.stack([t.argmax(-1) for t in logits],dim=1).cpu().tolist()


def run(model, seeds, lengths, n, rng, device):
    result=[]; failures=[]
    for length in lengths:
        init=[list(rng.choice(seeds)) for _ in range(n)]
        oracle=[x[:] for x in init]; roll=[x[:] for x in init]
        first_bad=[None]*n; conditional_good=0; total_steps=0
        schedule=make_schedule(init,n,length,rng,'all')
        for step,(ax,vals) in enumerate(schedule,1):
            next_oracle=[apply_operation(s,1,(a,0),(v,0)) for s,a,v in zip(oracle,ax,vals)]
            # Teacher-forced: test every step when starting from the correct state.
            conditional=forward_step(model,oracle,ax,vals,device)
            conditional_good+=sum(p==q for p,q in zip(conditional,next_oracle))
            total_steps+=n
            # Free rollout: feed model outputs back as its next input.
            roll=forward_step(model,roll,ax,vals,device)
            for i,(pred,truth) in enumerate(zip(roll,next_oracle)):
                if first_bad[i] is None and pred!=truth:
                    first_bad[i]=step
                    if len(failures)<25:
                        failures.append({'length':length,'sample':i,'first_bad_step':step,
                                         'initial':init[i],'axis':ax[i],'set_to':vals[i],
                                         'predicted':pred,'expected':truth})
            oracle=next_oracle
        axis_ok=sum(sum(a==b for a,b in zip(p,t)) for p,t in zip(roll,oracle))
        whole=sum(p==t for p,t in zip(roll,oracle))
        never=sum(x is None for x in first_bad)
        result.append({'chain_length':length,'examples':n,
                       'oracle_conditioned_step_exact':round(conditional_good/total_steps,6),
                       'free_rollout_final_axis':round(axis_ok/(n*12),6),
                       'free_rollout_final_exact':round(whole/n,6),
                       'never_diverged':round(never/n,6),
                       'first_failure_step_mean':round(sum(x for x in first_bad if x is not None)/max(1,n-never),3) if n!=never else None})
    return result,failures


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',default='operation_checkpoints/best_ops.pt')
    ap.add_argument('--data',default='grammar_training/test.jsonl')
    ap.add_argument('--out',default='chain_stress_report.json')
    ap.add_argument('--lengths',default='1,2,4,8,16,32,64')
    ap.add_argument('--examples',type=int,default=256)
    ap.add_argument('--seed',type=int,default=20261014)
    ap.add_argument('--device',default='cpu')
    args=ap.parse_args();torch.set_num_threads(2)
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    config=ck['config']; model=CrystalReentry(**config)
    model.load_state_dict(ck['model'],strict=True);model.to(args.device).eval()
    rows=read_records(Path(args.data))
    seeds=[r['targets']['indices'] for r in rows]
    for x in seeds: assert digits_from_index(index_from_digits(x))==x
    lengths=[int(v) for v in args.lengths.split(',')]
    if not lengths or any(v<1 for v in lengths) or args.examples<1: ap.error('positive lengths and examples required')
    metrics, failures=run(model,seeds,lengths,args.examples,random.Random(args.seed),args.device)
    report={'scope':'learned Induxion coordinate-edit chains',
            'checkpoint':args.checkpoint,'test_records':len(rows),'seed':args.seed,
            'model_parameters':sum(p.numel() for p in model.parameters()),
            'metrics':metrics,'example_failures':failures}
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    for row in metrics: print(json.dumps(row),flush=True)
    print('REPORT',out)

if __name__=='__main__':main()
