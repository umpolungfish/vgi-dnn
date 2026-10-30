import argparse, json
from pathlib import Path
import torch
from integrated_stress import load_ops, load_carrier, assess

def execute(original, candidate, carrier, lengths=(16,32,64,128,256), cases=128):
    torch.set_num_threads(1)
    old=load_ops(original); new=load_ops(candidate); c=load_carrier(carrier)
    results=[]
    for length in lengths:
        a=assess(old,c,cases,length,seed=13+length)
        b=assess(new,c,cases,length,seed=13+length)
        keys=('coordinate_step_accuracy','mixed_intermediate_accuracy','full_trajectory_exact','final_state_exact')
        results.append({'steps':length,'original':{k:a[k] for k in keys},'repaired':{k:b[k] for k in keys},'delta':{k:round(b[k]-a[k],6) for k in keys},'original_failed_cases':a['cases_with_intermediate_failure'],'repaired_failed_cases':b['cases_with_intermediate_failure']})
    return {'cases_per_length':cases,'results':results,'scope':'Paired deterministic mixed-operation rollouts.'}
