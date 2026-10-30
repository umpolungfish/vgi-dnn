#!/usr/bin/env python3
"""Exercise persisted graph-keyed re-entry and two sequential GPU updates."""
import argparse
import copy
import ctypes
import json
from pathlib import Path
import sys
import torch
from safetensors.torch import load_file,save_file
from composed_graph_reference import forward
from graph_reentry_bank import load,initial,store
from optimizer_state import split_state
from train_cuda_trajectories import adam_step,carrier,sha


def verify(root,runtime,dataset,state_path,out):
    torch.set_num_threads(1);sys.path.insert(0,str(root/'bob3ect/reentry_dnn'))
    from dtransformer_torch import DialetheicTransformerTorch
    build=json.loads((runtime/'build.json').read_text());checkpoint=runtime/'returned_checkpoint'
    receipt=json.loads((runtime/'trajectory_training_receipt.json').read_text())
    assert sha(checkpoint/'model.safetensors')==receipt['checkpoint_sha256']
    assert any(r['state_sha256']==sha(state_path) for r in receipt['training_graphs'])
    state=json.loads(state_path.read_text());row=state['graph']
    rows=[json.loads(s) for s in (dataset/'transitions.jsonl').open()]
    assert row in rows and row['split']=='train'
    vocabulary=json.loads((dataset/'vocabulary.json').read_text())
    lib=ctypes.CDLL(str(runtime/'model.so'));oracle=lib.ig_flow_gate
    oracle.argtypes=[ctypes.c_uint]*5;oracle.restype=ctypes.c_uint
    weights,optimizer=split_state(load_file(str(checkpoint/'model.safetensors')))
    clock=int(optimizer['clock'][0]);width=build['config']['d_model']
    bank=load(checkpoint,rows,sha(dataset/'transitions.jsonl'),sha(dataset/'vocabulary.json'),width,clock)
    store(bank,state,width,clock)
    out.mkdir(parents=True,exist_ok=False)
    (out/'initial_bank.json').write_text(json.dumps(bank)+'\n')
    def execute(devices):
        current=copy.deepcopy(weights);opt=copy.deepcopy(optimizer)
        carried=json.loads((out/'initial_bank.json').read_text());evidence=[]
        for index,device in enumerate(devices):
            model=DialetheicTransformerTorch(**build['config']).double().to(device);model.load_state_dict(current)
            saved=initial(carried,row,width,int(opt['clock'][0]));assert saved is not None
            result=forward(model,row,vocabulary,oracle,2,.25,initial=saved)
            before=float(result['loss'].detach());result['loss'].backward()
            adam_step(model,opt,build['parameters'],build['learning_rate'],build['stream_gradient_clip'])
            with torch.no_grad():
                after=forward(model,row,vocabulary,oracle,2,.25,initial=saved)
                reset=forward(model,row,vocabulary,oracle,2,.25)
            returned=carrier(after,row,device,int(opt['clock'][0]));store(carried,returned,width,int(opt['clock'][0]))
            # JSON roundtrip forces the next visit to consume serialized state.
            carried=json.loads(json.dumps(carried))
            current={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            evidence.append({'device':device,'loss_before':before,'loss_after':returned['loss'],
                'reset_difference':abs(returned['loss']-float(reset['loss'])),
                'initial_clock':saved['optimizer_clock'],'returned_clock':int(opt['clock'][0])})
        return current,opt,carried,evidence
    expected,eo,eb,ee=execute(['cpu','cpu'])
    actual,ao,ab,ae=execute(['cuda:0','cuda:1'])
    error=max(float((v.double()-actual[k].double()).abs().max()) for k,v in expected.items())
    optimizer_error=max(float((v.double()-ao[k].double()).abs().max()) for k,v in eo.items())
    memory_error=0.
    for identity in eb['states']:
        for key in ('memory_t','memory_f','probabilities'):
            a=eb['states'][identity]['state'][key];b=ab['states'][identity]['state'][key]
            memory_error=max(memory_error,max(abs(x-y) for xs,ys in zip(a,b) for x,y in zip(xs,ys)))
    assert max(error,optimizer_error,memory_error)<1e-8
    assert all(e['reset_difference']>1e-10 for e in ae)
    assert ae[1]['initial_clock']==ae[0]['returned_clock']
    save_file(actual|{'__optimizer.'+k:v for k,v in ao.items()},str(out/'model.safetensors'))
    (out/'returned_bank.json').write_text(json.dumps(ab)+'\n')
    report={'status':'passed','parameter_error':error,'optimizer_error':optimizer_error,'memory_error':memory_error,
        'observations':ae,'checkpoint_sha256':sha(checkpoint/'model.safetensors'),'source_state_sha256':sha(state_path),
        'scope':'persisted source-bound keyed bank re-enters same feedback graph across CPU/CPU and CUDA0/CUDA1 updates; reset control separate; production integration pending'}
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','runtime','dataset','state','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();verify(a.root.resolve(),a.runtime.resolve(),a.dataset.resolve(),a.state.resolve(),a.out.resolve())
