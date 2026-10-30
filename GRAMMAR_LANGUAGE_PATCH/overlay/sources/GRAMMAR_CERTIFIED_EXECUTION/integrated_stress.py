#!/usr/bin/env python3
"""Mixed composition stress: pretrained crystal operator + pretrained carrier kernel.

This is a *joint inference composition*, not a jointly optimized network or
native IMASM trace. Oracle after each step is computed independently.
"""
import argparse, json, random, sys
from pathlib import Path
import torch
from grammar_ops import CrystalReentry, RADICES, apply_operation
from carrier_net import CarrierKernel, bits, mask_from_bits
from dynamic_bridge import evaluate, split


def load_ops(path):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    model = CrystalReentry(**ck['config'])
    model.load_state_dict(ck['model'])
    model.eval()
    return model


def load_carrier(path):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    model = CarrierKernel()
    model.load_state_dict(ck['model_state'])
    model.eval()
    return model


def op_step(model, state, a, b, new_a, new_b, mode):
    vals = torch.tensor([state + [new_a, new_b]], dtype=torch.long)
    axes = torch.tensor([[a, b]], dtype=torch.long)
    mm = torch.tensor([mode], dtype=torch.long)
    with torch.no_grad():
        return [int(v.argmax(-1)[0]) for v in model(vals, axes, mm)]


def carrier_step(model, state, arity, flags):
    arms = split(state, arity)
    inp = torch.tensor([bits(a) + [int(f)] for a, f in zip(arms, flags)], dtype=torch.float32)
    with torch.no_grad():
        transformed = (model(inp[:, :4], inp[:, 4:]) > 0).int().tolist()
    fused = 0
    for arm in transformed: fused |= mask_from_bits(arm)
    return fused


def assess(ops, carrier, cases=256, steps=64, seed=13):
    rng = random.Random(seed)
    final_correct = 0; never_diverged = 0; intermediate_correct = 0
    crystal_ops = 0; crystal_ok = 0; carrier_ops = 0; carrier_ok = 0
    recovered_from_error = 0; first_failures=[]; examples=[]
    for case in range(cases):
        initial=[rng.randrange(r) for r in RADICES]
        crystal_truth=initial[:]; crystal_prediction=initial[:]
        carrier_truth=rng.randrange(16); carrier_prediction=carrier_truth
        first_failure=None; all_good=True
        for step in range(steps):
            if rng.randrange(2)==0:
                a=rng.randrange(12);b=(a+rng.randrange(1,12))%12
                v1=rng.randrange(RADICES[a]);v2=rng.randrange(RADICES[b]);mode=rng.choice((0,1,2))
                crystal_truth=apply_operation(crystal_truth,mode,(a,b),(v1,v2))
                crystal_prediction=op_step(ops,crystal_prediction,a,b,v1,v2,mode)
                crystal_ops+=1;crystal_ok+=(crystal_truth==crystal_prediction)
            else:
                arity=rng.choice((2,3)); flags=tuple(rng.randrange(2) for _ in range(arity))
                carrier_truth=evaluate(carrier_truth,arity,flags)['fused_mask']
                carrier_prediction=carrier_step(carrier,carrier_prediction,arity,flags)
                carrier_ops+=1;carrier_ok+=(carrier_truth==carrier_prediction)
            exact = crystal_truth==crystal_prediction and carrier_truth==carrier_prediction
            intermediate_correct+=exact
            if not exact:
                all_good=False
                if first_failure is None:first_failure=step+1
        final = crystal_truth==crystal_prediction and carrier_truth==carrier_prediction
        final_correct+=final;never_diverged+=all_good
        if first_failure is not None:
            first_failures.append(first_failure)
            recovered_from_error+=final
            if len(examples)<8:examples.append({'case':case,'first_failure_step':first_failure,'final_recovered':final})
    return {'cases':cases,'steps_per_case':steps,'total_steps':cases*steps,
        'coordinate_step_accuracy':round(crystal_ok/max(1,crystal_ops),6),
        'carrier_step_accuracy':round(carrier_ok/max(1,carrier_ops),6),
        'mixed_intermediate_accuracy':round(intermediate_correct/(cases*steps),6),
        'full_trajectory_exact':round(never_diverged/cases,6),
        'final_state_exact':round(final_correct/cases,6),
        'cases_with_intermediate_failure':len(first_failures),
        'cases_recovered_by_final':recovered_from_error,
        'first_failure_mean':round(sum(first_failures)/len(first_failures),3) if first_failures else None,
        'failure_examples':examples,
        'qualification':'pretrained coordinate and carrier neural operators with exact operational targets'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ops-checkpoint',default='../IMASM_GRAMMAR_DNN_OPS/operation_checkpoints/best_ops.pt')
    p.add_argument('--carrier-checkpoint',default='../GRAMMAR_CARRIER_LEARNER/carrier_checkpoints/best_carrier.pt')
    p.add_argument('--cases',type=int,default=256)
    p.add_argument('--lengths',nargs='+',type=int,default=[8,16,32,64,128,256])
    p.add_argument('--out',default='integrated_report.json')
    a=p.parse_args()
    torch.set_num_threads(1)
    ops=load_ops(a.ops_checkpoint);carrier=load_carrier(a.carrier_checkpoint)
    results=[]
    for length in a.lengths:
        item=assess(ops,carrier,a.cases,length,seed=13+length)
        print(json.dumps(item),flush=True);results.append(item)
    output={'ops_checkpoint':a.ops_checkpoint,'carrier_checkpoint':a.carrier_checkpoint,'results':results}
    Path(a.out).write_text(json.dumps(output,indent=2),encoding='utf-8')
    print('REPORT',a.out)
if __name__=='__main__': main()
