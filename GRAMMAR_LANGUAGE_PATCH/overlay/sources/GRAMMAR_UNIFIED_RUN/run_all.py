#!/usr/bin/env python3
"""One entrypoint: test, exact-frame certification, paired A/B, execution, attribution."""
import argparse, json, sys, unittest
from datetime import datetime, timezone
from pathlib import Path
import torch
from zeta_oracle import exact_certificate
from sic_gate import certificate
from integrated_stress import load_ops, load_carrier
from certified_execution import trial
from oracle_attribution import run as attribution_run
from paired_compare import execute as compare
from oracle_replay import export_corrections

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent

def find_checkpoint(explicit, candidates, label):
    if explicit:
        p=Path(explicit).expanduser().resolve()
        if not p.is_file():
            raise SystemExit(f'Missing {label} checkpoint: {p}')
        return p
    # Prefer the user's own trained checkpoints wherever they placed them.
    for relative in candidates:
        p=PARENT/relative
        if p.is_file(): return p.resolve()
    names={Path(c).name for c in candidates}
    roots=[PARENT, Path.home()/'imsgct']
    found=[]
    for root in roots:
        if not root.is_dir(): continue
        for filename in names:
            found.extend(p for p in root.glob('**/'+filename)
                         if p.is_file() and 'bundled_checkpoints' not in p.parts)
    if found:
        # Prefer repaired weights for ops, original for comparison.
        key=lambda p: (('GRAMMAR_COORDINATE_REPAIR' not in p.parts) if label=='ops' else
                       ('IMASM_GRAMMAR_DNN_OPS' not in p.parts), str(p))
        return sorted(set(found),key=key)[0].resolve()
    fallback=HERE/'bundled_checkpoints'/('best_carrier.pt' if label=='carrier' else 'best_ops.pt')
    if fallback.is_file():
        print('CHECKPOINT_FALLBACK',json.dumps({'role':label,'path':str(fallback),
              'note':'Packaged baseline weights; not your locally trained checkpoint'}),flush=True)
        return fallback.resolve()
    raise SystemExit('Missing '+label+' checkpoint. Supply --'+label+'-checkpoint PATH')

def tests():
    suite=unittest.defaultTestLoader.discover(str(HERE), pattern='test_*.py')
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():raise SystemExit('Tests failed; skipping evaluation')
    return {'run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped)}

def main():
    parser=argparse.ArgumentParser(description='All-in-one Grammar evaluation and oracle attribution')
    parser.add_argument('--ops-checkpoint',type=str,help='Override repaired coordinate checkpoint')
    parser.add_argument('--original-checkpoint',type=str,help='Override original coordinate checkpoint')
    parser.add_argument('--carrier-checkpoint',type=str,help='Override carrier checkpoint')
    parser.add_argument('--cases',type=int,default=128)
    parser.add_argument('--steps',type=int,default=256)
    parser.add_argument('--quick',action='store_true',help='Run a fast 8-case x 16-step smoke test')
    parser.add_argument('--skip-compare',action='store_true',help='Skip paired original versus repaired rollouts')
    a=parser.parse_args()
    if a.cases<1 or a.steps<1:parser.error('cases and steps must be positive')
    cases=8 if a.quick else a.cases
    steps=16 if a.quick else a.steps
    torch.set_num_threads(1)
    results={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'Exact rational Gram identities, coordinate and carrier operations, neural evaluation, and oracle attribution.'}
    results['tests']=tests()
    results['sic_zeta']={'exact_gram_inverses':[exact_certificate(d) for d in (2,4,12)]}
    print('SIC_ZETA',json.dumps({'dimensions':[2,4,12],'exact_left_inverses':all(c['left_inverse_exact'] for c in results['sic_zeta']['exact_gram_inverses'])}),flush=True)
    repaired=find_checkpoint(a.ops_checkpoint,['GRAMMAR_COORDINATE_REPAIR/coordinate_checkpoints/candidate_ops.pt','IMASM_GRAMMAR_DNN_OPS/operation_checkpoints/best_ops.pt'],'ops')
    carrier=find_checkpoint(a.carrier_checkpoint,['GRAMMAR_CARRIER_LEARNER/carrier_checkpoints/best_carrier.pt'],'carrier')
    results['checkpoints']={'ops':str(repaired),'carrier':str(carrier)}
    results['checkpoint_sources']={'ops':'bundled_baseline' if 'bundled_checkpoints' in repaired.parts else 'local', 'carrier':'bundled_baseline' if 'bundled_checkpoints' in carrier.parts else 'local'}
    ops_model=load_ops(repaired); carrier_model=load_carrier(carrier)
    results['certified_execution']=trial(ops_model,carrier_model,cases,steps,159)
    print('CERTIFIED',json.dumps({k:v for k,v in results['certified_execution'].items() if k!='failure_examples'}),flush=True)
    results['oracle_attribution']=attribution_run(ops_model,carrier_model,cases,steps,159)
    print('ATTRIBUTION',json.dumps({k:v for k,v in results['oracle_attribution'].items() if k!='corrections'}),flush=True)
    results['oracle_replay']=export_corrections(results['oracle_attribution'],HERE/'oracle_corrections.jsonl')
    print('REPLAY',json.dumps(results['oracle_replay']),flush=True)
    if not a.skip_compare:
        original=find_checkpoint(a.original_checkpoint,['IMASM_GRAMMAR_DNN_OPS/operation_checkpoints/best_ops.pt'],'original')
        results['checkpoints']['original']=str(original)
        results['checkpoint_sources']['original']='bundled_baseline' if 'bundled_checkpoints' in original.parts else 'local'
        if original == repaired:
            results['paired_comparison']={'skipped':'Original and selected coordinate checkpoint are the same file'}
            print('PAIRED_SKIPPED same checkpoint used for both roles',flush=True)
        else:
            lengths=(8,16) if a.quick else (8,16,32,64,128,256)
            results['paired_comparison']=compare(original,repaired,carrier,lengths,max(8,cases))
            for r in results['paired_comparison']['results']:print('PAIRED',json.dumps(r),flush=True)
    report=HERE/'unified_report.json'
    report.write_text(json.dumps(results,indent=2,default=str)+'\n',encoding='utf-8')
    print('REPORT',report,flush=True)

if __name__=='__main__':main()
