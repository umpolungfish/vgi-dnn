#!/usr/bin/env python3
"""SIC-frame-certified mixed inference with explicit transition oracle corrections.

The exact d=12/d=2 frame checks certify representation/reconstruction, not
semantic correctness. Separate operational oracles determine transition targets.
"""
import argparse, json, random
from fractions import Fraction as F
from pathlib import Path
import torch
from integrated_stress import load_ops, load_carrier, op_step, carrier_step
from grammar_ops import RADICES, apply_operation
from dynamic_bridge import evaluate
from zeta_oracle import exact_certificate, split, fuse


def frame_check(address, carrier):
    if len(address)!=12 or any(type(v) is not int or v<0 or v>=r for v,r in zip(address,RADICES)):
        return False
    if type(carrier) is not int or not 0<=carrier<16: return False
    # Lossless embedding into 144-dim rational coefficient domain.
    c=[F(x) for x in address]+[F(0)]*132
    b=[F((carrier>>i)&1) for i in range(4)]
    return fuse(split(c,12),12)==c and fuse(split(b,2),2)==b


def trial(ops, carrier_net, cases, steps, seed):
    rng=random.Random(seed)
    counts={'coordinate_proposals':0,'coordinate_accepted':0,'coordinate_corrected':0,
            'carrier_proposals':0,'carrier_accepted':0,'carrier_corrected':0,
            'frame_reconstruction_failures':0,'raw_full_trajectory_exact':0,
            'raw_final_exact':0,'gated_full_trajectory_exact':0,'gated_final_exact':0}
    failures=[]
    for case in range(cases):
        truth=[rng.randrange(r) for r in RADICES]; raw=truth[:]; gated=truth[:]
        ct=rng.randrange(16);cr=ct;cg=ct
        raw_all=True;gate_all=True
        for tick in range(steps):
            if rng.randrange(2)==0:
                a=rng.randrange(12);b=(a+rng.randrange(1,12))%12
                u=rng.randrange(RADICES[a]);v=rng.randrange(RADICES[b]);mode=rng.randrange(3)
                # Oracle is independently derived from current truth / gated inputs.
                truth=apply_operation(truth,mode,(a,b),(u,v))
                proposed_raw=op_step(ops,raw,a,b,u,v,mode)
                proposed_gate=op_step(ops,gated,a,b,u,v,mode)
                target_gate=apply_operation(gated,mode,(a,b),(u,v))
                counts['coordinate_proposals']+=1
                if proposed_gate==target_gate:
                    gated=proposed_gate;counts['coordinate_accepted']+=1
                else:
                    gated=target_gate;counts['coordinate_corrected']+=1
                    if len(failures)<12:failures.append({'case':case,'step':tick+1,'kind':'coordinate',
                         'mode':mode,'axes':[a,b],'values':[u,v],
                         'proposed':proposed_gate,'expected':target_gate})
                raw=proposed_raw
            else:
                arity=rng.choice((2,3)); flags=tuple(rng.randrange(2) for _ in range(arity))
                ct=evaluate(ct,arity,flags)['fused_mask']
                proposed_raw=carrier_step(carrier_net,cr,arity,flags)
                proposed_gate=carrier_step(carrier_net,cg,arity,flags)
                target_gate=evaluate(cg,arity,flags)['fused_mask']
                counts['carrier_proposals']+=1
                if proposed_gate==target_gate:
                    cg=proposed_gate;counts['carrier_accepted']+=1
                else:
                    cg=target_gate;counts['carrier_corrected']+=1
                    if len(failures)<12:failures.append({'case':case,'step':tick+1,'kind':'carrier',
                        'arity':arity,'flags':list(flags),'proposed':proposed_gate,'expected':target_gate})
                cr=proposed_raw
            if not frame_check(gated,cg):counts['frame_reconstruction_failures']+=1
            raw_all &= (raw==truth and cr==ct)
            gate_all &= (gated==truth and cg==ct)
        counts['raw_full_trajectory_exact']+=int(raw_all)
        counts['raw_final_exact']+=int(raw==truth and cr==ct)
        counts['gated_full_trajectory_exact']+=int(gate_all)
        counts['gated_final_exact']+=int(gated==truth and cg==ct)
    return {'cases':cases,'steps':steps,'counts':counts,
            'raw_final_accuracy':counts['raw_final_exact']/cases,
            'raw_trajectory_accuracy':counts['raw_full_trajectory_exact']/cases,
            'gated_final_accuracy':counts['gated_final_exact']/cases,
            'gated_trajectory_accuracy':counts['gated_full_trajectory_exact']/cases,
            'failure_examples':failures}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ops-checkpoint',default='../GRAMMAR_COORDINATE_REPAIR/coordinate_checkpoints/candidate_ops.pt')
    p.add_argument('--carrier-checkpoint',default='../GRAMMAR_CARRIER_LEARNER/carrier_checkpoints/best_carrier.pt')
    p.add_argument('--cases',type=int,default=128)
    p.add_argument('--steps',type=int,default=256)
    p.add_argument('--seed',type=int,default=159)
    p.add_argument('--out',default='certified_execution_report.json')
    a=p.parse_args()
    torch.set_num_threads(1)
    exact=[exact_certificate(d) for d in (2,4,12)]
    ops=load_ops(a.ops_checkpoint); carrier=load_carrier(a.carrier_checkpoint)
    result=trial(ops,carrier,a.cases,a.steps,a.seed)
    result['exact_frame_dimensions']=[x['dimension'] for x in exact]
    result['scope']='Exact SIC frame reconstruction; exact operation-oracle corrections; independent raw and corrected accuracy.'
    Path(a.out).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('RESULT',json.dumps({k:v for k,v in result.items() if k!='failure_examples'}))
    print('REPORT',a.out)
if __name__=='__main__':main()
