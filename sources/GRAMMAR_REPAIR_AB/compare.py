#!/usr/bin/env python3
"""Paired original-versus-repaired mixed rollout with identical random streams."""
import argparse, json, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'GRAMMAR_INTEGRATED_V1'))
try:
    import torch
    from integrated_stress import load_ops, load_carrier, assess
except ImportError as e:
    raise SystemExit(f'Missing dependency or sibling GRAMMAR_INTEGRATED_V1: {e}')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--original',default=str(ROOT/'IMASM_GRAMMAR_DNN_OPS/operation_checkpoints/best_ops.pt'))
    p.add_argument('--candidate',default=str(ROOT/'GRAMMAR_COORDINATE_REPAIR/coordinate_checkpoints/candidate_ops.pt'))
    p.add_argument('--carrier',default=str(ROOT/'GRAMMAR_CARRIER_LEARNER/carrier_checkpoints/best_carrier.pt'))
    p.add_argument('--lengths',nargs='+',type=int,default=[8,16,32,64,128,256])
    p.add_argument('--cases',type=int,default=256)
    p.add_argument('--out',default='repair_ab_report.json')
    args=p.parse_args()
    for path in [args.original,args.candidate,args.carrier]:
        if not Path(path).is_file():
            raise SystemExit(f'Checkpoint missing: {path}')
    if args.cases < 1 or not args.lengths or min(args.lengths)<1:
        p.error('cases and lengths must be positive')
    torch.set_num_threads(1)
    base=load_ops(args.original); candidate=load_ops(args.candidate); carrier=load_carrier(args.carrier)
    results=[]
    for length in args.lengths:
        seed=13+length
        b=assess(base,carrier,args.cases,length,seed=seed)
        c=assess(candidate,carrier,args.cases,length,seed=seed)
        keys=['coordinate_step_accuracy','mixed_intermediate_accuracy','full_trajectory_exact','final_state_exact']
        delta={k:round(c[k]-b[k],6) for k in keys}
        result={'steps':length,'original':{k:b[k] for k in keys},'candidate':{k:c[k] for k in keys},'candidate_minus_original':delta,'original_failed_cases':b['cases_with_intermediate_failure'],'candidate_failed_cases':c['cases_with_intermediate_failure']}
        results.append(result)
        print('COMPARE',json.dumps(result),flush=True)
    report={'method':'Paired deterministic seeds, same carrier checkpoint, separate rollouts from same initial states','original':args.original,'candidate':args.candidate,'carrier':args.carrier,'cases_per_length':args.cases,'results':results,'note':'Paired coordinate/carrier operational-oracle evaluation.'}
    Path(args.out).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('REPORT',args.out)
if __name__=='__main__':main()
