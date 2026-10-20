#!/usr/bin/env python3
"""Compare executing IMASM learner parameters and Adam memory with Torch."""
import argparse,ctypes,json,sys
from pathlib import Path
import torch
from optimizer_state import load_model
from retained_conditioning import evaluate
from torch_reference_optimizer import make_optimizer


def verify(root,runtime):
    torch.set_num_threads(1);sys.path.insert(0,str(root/'bob3ect/reentry_dnn'))
    from dtransformer_torch import DialetheicTransformerTorch
    build=json.loads((runtime/'build.json').read_text());assert build['optimizer']=='adam'
    model=DialetheicTransformerTorch(**build['config']).double();model.load_state_dict(load_model(Path(build['parent'])/'model.safetensors'))
    optimizer=make_optimizer(model,build);lib=ctypes.CDLL(str(runtime/'model.so'));lib.vgi_retained_reset()
    for name,count in [('vgi_retained_token',2),('vgi_retained_update',6)]:
        getattr(lib,name).argtypes=[ctypes.c_long]*count;getattr(lib,name).restype=ctypes.c_long
    for name,count in [('vgi_full_parameter',1),('vgi_optimizer_observe',2)]:
        getattr(lib,name).argtypes=[ctypes.c_long]*count;getattr(lib,name).restype=ctypes.c_double
    lib.ig_flow_gate.argtypes=[ctypes.c_uint]*5;lib.ig_flow_gate.restype=ctypes.c_uint
    cases=[(0,0,seed,0,1) for seed in (0,3,15,5,10)] + [(6,15,3,slot,3) for slot in range(3)] + [(3,5,15,0,1),(9,15,3,0,1)]
    updates=0
    for opcode,input_bits,seed,slot,fan in cases:
        bits=lib.ig_flow_gate(opcode,input_bits,seed,slot,fan);assert bits<16
        query=[256,71,opcode,266+input_bits,266+seed,fan,slot,257]
        for i,token in enumerate(query):assert lib.vgi_retained_token(i,token)==0
        assert lib.vgi_retained_update(8,0,2,bits,1,1)>=0
        optimizer.zero_grad();result=evaluate(model,query,[],context=16,kind=2,bits=bits)
        for loss in result['loss_windows']:loss.backward()
        for p in model.parameters():p.grad.div_(result['labels'])
        norm=float(torch.sqrt(sum((p.grad*p.grad).sum() for p in model.parameters())))
        scale=min(1.,build['stream_gradient_clip']/norm) if norm and build['stream_gradient_clip']>0 else 1.
        for p in model.parameters():p.grad.mul_(scale)
        optimizer.step();updates+=1
    errors={'parameters':0.,'first_moments':0.,'second_moments':0.};position=0;parameters=dict(model.named_parameters())
    for entry in build['parameters']:
        p=parameters[entry['key']];n=entry['size']
        for kind,key,expected in [(None,'parameters',p.detach()),(1,'first_moments',optimizer.state[p]['exp_avg']),(2,'second_moments',optimizer.state[p]['exp_avg_sq'])]:
            actual=torch.tensor([lib.vgi_full_parameter(i) if kind is None else lib.vgi_optimizer_observe(kind,i) for i in range(position,position+n)],dtype=torch.double)
            errors[key]=max(errors[key],float((actual-expected.flatten()).abs().max()))
        position+=n
    assert max(errors.values())<1e-8,errors
    clock=build['optimizer_initial_clock']+updates;assert lib.vgi_optimizer_observe(0,0)==clock
    powers=[lib.vgi_optimizer_observe(3,i) for i in range(2)]
    assert max(abs(powers[i]-beta**clock) for i,beta in enumerate((.9,.999)))<1e-12
    report={'status':'passed','updates':updates,'clock':clock,'maximum_errors':errors,'bias_powers':powers,
            'comparison':'all model parameters, both Adam moment vectors, global clock, and bias powers'}
    (runtime/'adam_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','runtime'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();verify(a.root.resolve(),a.runtime.resolve())
