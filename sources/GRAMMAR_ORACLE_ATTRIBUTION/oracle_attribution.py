#!/usr/bin/env python3
"""Distinguish invertible representation from semantic correctness, with full replay traces."""
import argparse
import json
import random
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / 'GRAMMAR_CERTIFIED_EXECUTION'
if not BASE.is_dir():
    raise SystemExit(f'Missing sibling project {BASE}; extract alongside GRAMMAR_CERTIFIED_EXECUTION')
sys.path.insert(0, str(BASE))

import torch
from certified_execution import frame_check
from integrated_stress import load_ops, load_carrier, op_step, carrier_step
from grammar_ops import RADICES, apply_operation
from dynamic_bridge import evaluate
from zeta_oracle import exact_certificate


def run(ops, carrier, cases=128, steps=256, seed=159):
    rng = random.Random(seed)
    stats = {'coordinate_proposals':0, 'coordinate_wrong':0, 'carrier_proposals':0,
             'carrier_wrong':0, 'invalid_frame_proposals':0,
             'semantically_wrong_but_frame_valid':0, 'raw_trajectories_exact':0,
             'raw_final_exact':0, 'gated_trajectories_exact':0, 'gated_final_exact':0,
             'recovered_raw_trajectories':0}
    corrections=[]
    for case in range(cases):
        truth_addr=[rng.randrange(r) for r in RADICES]
        raw_addr=truth_addr[:]; gated_addr=truth_addr[:]
        truth_c=rng.randrange(16); raw_c=truth_c; gated_c=truth_c
        raw_ok=True; gated_ok=True
        for tick in range(1,steps+1):
            if rng.randrange(2)==0:
                a=rng.randrange(12); b=(a+rng.randrange(1,12))%12
                u=rng.randrange(RADICES[a]); v=rng.randrange(RADICES[b]); mode=rng.randrange(3)
                before_truth=truth_addr[:]
                truth_addr=apply_operation(truth_addr,mode,(a,b),(u,v))
                raw_addr=op_step(ops,raw_addr,a,b,u,v,mode)
                before_gated=gated_addr[:]
                proposed=op_step(ops,gated_addr,a,b,u,v,mode)
                expected=apply_operation(gated_addr,mode,(a,b),(u,v))
                stats['coordinate_proposals']+=1
                kind='coordinate'; info={'mode':mode,'axes':[a,b],'values':[u,v]}
                before={'address':before_gated,'carrier':gated_c}
                proposal={'address':proposed,'carrier':gated_c}
                target={'address':expected,'carrier':gated_c}
                valid=frame_check(proposed,gated_c)
                if proposed!=expected:
                    stats['coordinate_wrong']+=1
                    gated_addr=expected
                else: gated_addr=proposed
            else:
                arity=rng.choice((2,3)); flags=tuple(rng.randrange(2) for _ in range(arity))
                truth_c=evaluate(truth_c,arity,flags)['fused_mask']
                raw_c=carrier_step(carrier,raw_c,arity,flags)
                before_gated=gated_c
                proposed=carrier_step(carrier,gated_c,arity,flags)
                expected=evaluate(gated_c,arity,flags)['fused_mask']
                stats['carrier_proposals']+=1
                kind='carrier';info={'arity':arity,'flags':list(flags)}
                before={'address':gated_addr[:],'carrier':before_gated}
                proposal={'address':gated_addr[:],'carrier':proposed}
                target={'address':gated_addr[:],'carrier':expected}
                valid=frame_check(gated_addr,proposed)
                if proposed!=expected:
                    stats['carrier_wrong']+=1
                    gated_c=expected
                else:gated_c=proposed
            correct=proposal==target
            if not valid:stats['invalid_frame_proposals']+=1
            if not correct and valid:stats['semantically_wrong_but_frame_valid']+=1
            if not correct:
                corrections.append({'case':case,'step':tick,'kind':kind,'operation':info,
                                    'before':before,'proposed':proposal,'oracle_target':target,
                                    'frame_valid':valid,'oracle_corrected':True})
            raw_ok &= (raw_addr==truth_addr and raw_c==truth_c)
            gated_ok &= (gated_addr==truth_addr and gated_c==truth_c)
        stats['raw_trajectories_exact']+=int(raw_ok)
        stats['raw_final_exact']+=int(raw_addr==truth_addr and raw_c==truth_c)
        stats['gated_trajectories_exact']+=int(gated_ok)
        stats['gated_final_exact']+=int(gated_addr==truth_addr and gated_c==truth_c)
        stats['recovered_raw_trajectories']+=int(not raw_ok and raw_addr==truth_addr and raw_c==truth_c)
    total=stats['coordinate_proposals']+stats['carrier_proposals']
    return {'settings':{'cases':cases,'steps':steps,'seed':seed},'counts':stats,
            'raw_trajectory_accuracy':stats['raw_trajectories_exact']/cases,
            'raw_final_accuracy':stats['raw_final_exact']/cases,
            'frame_detection_rate_of_wrong_proposals':(
                (stats['coordinate_wrong']+stats['carrier_wrong']-stats['semantically_wrong_but_frame_valid'])
                /max(1,stats['coordinate_wrong']+stats['carrier_wrong'])),
            'oracle_correction_count':len(corrections),
            'corrections':corrections,
            'scope':'Frame: exact coordinate reconstruction. Transition oracle: exact coordinate and carrier operations.'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ops-checkpoint',default='../GRAMMAR_COORDINATE_REPAIR/coordinate_checkpoints/candidate_ops.pt')
    p.add_argument('--carrier-checkpoint',default='../GRAMMAR_CARRIER_LEARNER/carrier_checkpoints/best_carrier.pt')
    p.add_argument('--cases',type=int,default=128)
    p.add_argument('--steps',type=int,default=256)
    p.add_argument('--seed',type=int,default=159)
    p.add_argument('--out',default='oracle_attribution_report.json')
    args=p.parse_args()
    if args.cases<1 or args.steps<1:p.error('cases and steps must be positive')
    torch.set_num_threads(1)
    frames=[exact_certificate(d) for d in (2,4,12)]
    result=run(load_ops(args.ops_checkpoint),load_carrier(args.carrier_checkpoint),args.cases,args.steps,args.seed)
    result['verified_gram_dimensions']=[f['dimension'] for f in frames]
    Path(args.out).write_text(json.dumps(result,indent=2)+'\n')
    print('ATTRIBUTION',json.dumps({k:v for k,v in result.items() if k!='corrections'}))
    print('REPLAYABLE_CORRECTIONS',len(result['corrections']))
    print('REPORT',args.out)
if __name__=='__main__':main()
