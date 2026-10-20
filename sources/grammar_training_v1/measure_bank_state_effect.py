#!/usr/bin/env python3
"""Distinguish bank inputs supplied from observed effects on executing state."""
import argparse
import ctypes
import json
from pathlib import Path
import sys
import torch
from composed_graph_reference import forward
from graph_reentry_bank import load,initial
from optimizer_state import load_model
from train_cuda_trajectories import sha


def measure(root,runtime,dataset,out):
    torch.set_num_threads(1);sys.path.insert(0,str(root/'bob3ect/reentry_dnn'))
    from dtransformer_torch import DialetheicTransformerTorch
    build=json.loads((runtime/'build.json').read_text());parent=Path(build['parent'])
    report=json.loads((runtime/'trajectory_training_receipt.json').read_text())
    assert sha(parent/'model.safetensors')==report['parent_checkpoint_sha256']
    model=DialetheicTransformerTorch(**build['config']).double();model.load_state_dict(load_model(parent/'model.safetensors'))
    rows=[json.loads(s) for s in (dataset/'transitions.jsonl').open()]
    bank=load(parent,rows,sha(dataset/'transitions.jsonl'),sha(dataset/'vocabulary.json'),model.d,build['optimizer_initial_clock'])
    selected=[json.loads(s) for s in (runtime/'training_graphs.jsonl').open()]
    vocabulary=json.loads((dataset/'vocabulary.json').read_text());lib=ctypes.CDLL(str(runtime/'model.so'))
    oracle=lib.ig_flow_gate;oracle.argtypes=[ctypes.c_uint]*5;oracle.restype=ctypes.c_uint
    observations=[]
    with torch.no_grad():
        for row in selected:
            state=initial(bank,row,model.d,build['optimizer_initial_clock'])
            if state is None:continue
            carried=forward(model,row,vocabulary,oracle,report['passes'],report['local_weight'],initial=state)
            reset=forward(model,row,vocabulary,oracle,report['passes'],report['local_weight'])
            memory=max(float((a-b).abs().max()) for pa,pb in zip(carried['memory'],reset['memory']) for a,b in zip(pa,pb))
            probability=max(float((a-b).abs().max()) for a,b in zip(carried['probabilities'],reset['probabilities']))
            observations.append({'id':row['id'],'family':row['graph']['family'],
                'prior_edge_read':any(a>=b for a,b in row['graph']['edges']),
                'loss_difference':abs(float(carried['loss'])-float(reset['loss'])),
                'memory_difference':memory,'probability_difference':probability,
                'carried_return':carried['returned'],'reset_return':reset['returned']})
    result={'checkpoint_sha256':report['parent_checkpoint_sha256'],'bank_inputs_supplied':len(observations),
        'state_sensitive_graphs':sum(max(r['memory_difference'],r['probability_difference'])>1e-10 for r in observations),
        'return_sensitive_graphs':sum(r['carried_return']!=r['reset_return'] for r in observations),
        'observations':observations,'scope':'same frozen pre-stage model and source graph for carried-state/reset controls; does not replay intermediate production model updates'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='observations'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','runtime','dataset','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();measure(a.root.resolve(),a.runtime.resolve(),a.dataset.resolve(),a.out.resolve())
