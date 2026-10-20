#!/usr/bin/env python3
"""Compare every full-network gradient/update and capture Vox's learned state."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


def verify(root, runtime):
    torch.set_num_threads(1)
    build=json.loads((runtime/'build.json').read_text());parent=Path(build['parent'])
    state=load_file(str(parent/'model.safetensors'));cfg=build['config']
    policy=build.get('objective')=='policy';loss_bits=8 if policy else 4;loss_base=261 if policy else 259
    tokens=build.get('probe_tokens',json.loads((parent/'execution_input.json').read_text())['tokens'])
    sys.path.insert(0,str(root/'bob3ect/reentry_dnn'))
    from dtransformer_torch import DialetheicTransformerTorch,logit
    lib=ctypes.CDLL(str((runtime/'model.so').resolve()))
    lib.vgi_full_learn.argtypes=[ctypes.c_long]*4;lib.vgi_full_learn.restype=ctypes.c_long
    for name in ('vgi_full_gradient','vgi_full_parameter'):
        getattr(lib,name).argtypes=[ctypes.c_long];getattr(lib,name).restype=ctypes.c_double

    def reference(bits,steps,unroll):
        model=DialetheicTransformerTorch(**cfg).double();model.load_state_dict(state)
        optimizer=torch.optim.SGD(model.parameters(),lr=build.get('learning_rate',.001))
        target=torch.tensor([(bits>>k)&1 for k in range(loss_bits)],dtype=torch.double)
        initial_gradients={};before=0
        for step in range(steps+1):
            t,f=model.embed(torch.tensor([tokens]))
            for block in model.blocks:t,f=block(t,f)
            for _ in range(unroll):t,f=model.reentry_block(t,f)
            sup,ref=model.head(*[logit(z) for z in model.final_norm(t,f)])
            selected=torch.stack([(ref if k%2 else sup)[0,-1,loss_base+k//2] for k in range(loss_bits)])
            loss=F.binary_cross_entropy_with_logits(selected,target)
            if step==0:before=float(loss.detach())
            if step<steps or step==0:
                optimizer.zero_grad();loss.backward()
                if step==0:
                    initial_gradients={name:parameter.grad.detach().clone()
                                       for name,parameter in model.named_parameters()}
                if step<steps:optimizer.step()
        return model,initial_gradients,before,float(loss.detach())

    native=[]
    cases=[(0,1,0),(3,1,0),(15,2,0),(3,1,1)] if not policy else [
        (0,1,0),(255,1,0),(build['process_update']['bits'],2,0),(build['process_update']['bits'],1,1)]
    for bits,steps,unroll in cases:
        model,gradients,before,after=reference(bits,steps,unroll)
        returned=lib.vgi_full_learn(bits,steps,unroll,1)
        if returned<0:raise RuntimeError(f'native tape failed: {returned}')
        if abs(returned/1e6-after)>1e-6 or after>=before:
            raise RuntimeError('native loss/update mismatch')
        index=0;checks=[]
        parameters=dict(model.named_parameters())
        for entry in build['parameters']:
            name,size=entry['key'],entry['size']
            actual_gradient=torch.tensor([lib.vgi_full_gradient(i) for i in range(index,index+size)],dtype=torch.double)
            actual_parameter=torch.tensor([lib.vgi_full_parameter(i) for i in range(index,index+size)],dtype=torch.double)
            expected_gradient=gradients[name].flatten();expected_parameter=parameters[name].detach().flatten()
            ge=float((actual_gradient-expected_gradient).abs().max())
            pe=float((actual_parameter-expected_parameter).abs().max())
            if not torch.allclose(actual_gradient,expected_gradient,atol=1e-8,rtol=1e-5):
                raise RuntimeError(f'gradient mismatch {name}: {ge}')
            if not torch.allclose(actual_parameter,expected_parameter,atol=1e-8,rtol=1e-5):
                raise RuntimeError(f'parameter mismatch {name}: {pe}')
            checks.append({'tensor':name,'gradient_max_abs_error':ge,'parameter_max_abs_error':pe,
                           'nonzero_gradient_elements':int((actual_gradient!=0).sum())})
            index+=size
        row={'bits':bits,'steps':steps,'continuous_reentry_unroll':unroll,
             'loss_before':before,'loss_after':after,'tensors':checks,
             'tape_nodes':lib.vgi_full_learn(bits,steps,unroll,3)}
        native.append(row);print(json.dumps({k:v for k,v in row.items() if k!='tensors'}),flush=True)
    (runtime/'native_gradient_verification.json').write_text(json.dumps(native,indent=2)+'\n')
    # Process entry executes a complete update and emits its entire learned state.
    update=build['process_update']
    model,_,before,after=reference(update['bits'],update['steps'],update['continuous_reentry_unroll'])
    command=[str(root/'Vox/target/release/vox'),'run',str(runtime/'model.imasm')]
    env=dict(os.environ);env['VOX_STEPS']=str(max(1000000000,update['steps']*100000000))
    print('Executing full-network update and checkpoint return through Vox',flush=True)
    result=subprocess.run(command,capture_output=True,timeout=300,env=env)
    (runtime/'process.log').write_bytes(result.stderr)
    if result.returncode:raise RuntimeError(result.stderr.decode(errors='replace'))
    if len(result.stdout)<8:raise RuntimeError('no checkpoint returned')
    header_len=struct.unpack('<Q',result.stdout[:8])[0]
    header=json.loads(result.stdout[8:8+header_len]);length=8+header_len+max(
        value['data_offsets'][1] for key,value in header.items() if key!='__metadata__')
    if length!=len(result.stdout):raise RuntimeError('checkpoint boundary mismatch')
    output=runtime/'updated_checkpoint';output.mkdir(exist_ok=False)
    path=output/'model.safetensors';path.write_bytes(result.stdout)
    observed=load_file(str(path));expected=model.state_dict();checks=[]
    if set(observed)!=set(expected):raise RuntimeError('checkpoint tensor set mismatch')
    for name in observed:
        a,b=observed[name],expected[name]
        error=float((a.double()-b.double()).abs().max())
        if not torch.allclose(a.double(),b.double(),atol=1e-8,rtol=1e-5):
            raise RuntimeError(f'VM checkpoint mismatch: {name}: {error}')
        checks.append({'tensor':name,'max_abs_error':error,'changed_from_parent':not torch.equal(a.double(),state[name].double())})
    for filename in ('config.json','tokenizer.json'):
        shutil.copyfile(parent/filename,output/filename)
    with torch.no_grad():support,refute=model(torch.tensor([tokens]))
    original_probe=json.loads((parent/'execution_input.json').read_text())
    (output/'execution_input.json').write_text(json.dumps({'tokens':tokens,'source_id':original_probe['source_id'],
        'support':support[0].tolist(),'refute':refute[0].tolist(),
        'evidence':'reference_forward_from_returned_full_network_checkpoint'},indent=2)+'\n')
    report={'status':'passed','native':native,'vm_command':command,'vm_step_budget':env['VOX_STEPS'],
            'vm_checkpoint_tensors':checks,'loss_before':before,'loss_after':after,
            'module_sha256':hashlib.sha256((runtime/'model.imasm').read_bytes()).hexdigest(),
            'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'scope':'full-network scalar reverse mode and SGD; continuous tied re-entry; quantized settlement evaluated after reload'}
    (runtime/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':'passed','native_gradient_cases':len(native),
                      'vm_checkpoint_tensors':len(checks),'loss_before':before,'loss_after':after,
                      'changed_tensors':sum(c['changed_from_parent'] for c in checks)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True)
    a=p.parse_args();verify(a.root.resolve(),a.runtime.resolve())
