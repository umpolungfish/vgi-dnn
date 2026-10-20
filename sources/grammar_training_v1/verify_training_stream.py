#!/usr/bin/env python3
"""Verify consecutive clipped SGD updates and held-prefix return through Vox."""
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

from build_training_stream import MAGIC,pack


def verify(root,runtime,stream_dir):
    torch.set_num_threads(1)
    build=json.loads((runtime/'build.json').read_text());parent=Path(build['parent']);cfg=build['config']
    frames=[json.loads(l) for l in (stream_dir/'frames.jsonl').open()]
    wire=(stream_dir/'train.vgis').read_bytes();assert wire[:8]==MAGIC
    assert struct.unpack('<I',wire[8:12])[0]==len(frames)
    assert wire[12:]==b''.join(pack(frame) for frame in frames)
    for frame in frames:assert hashlib.sha256(pack(frame)).hexdigest()==frame['wire_sha256']
    if any(len(frame['tokens'])!=build['context_length'] for frame in frames):raise ValueError('context mismatch')
    if any(frame['steps']!=1 or frame['unroll']!=0 for frame in frames):
        raise ValueError('this independent fixture checks one update per frame without recurrent unroll')
    sys.path.insert(0,str(root/'bob3ect/reentry_dnn'))
    from dtransformer_torch import DialetheicTransformerTorch
    state=load_file(str(parent/'model.safetensors'));model=DialetheicTransformerTorch(**cfg).double();model.load_state_dict(state)
    optimizer=torch.optim.SGD(model.parameters(),lr=build['learning_rate'])
    lib=ctypes.CDLL(str((runtime/'model.so').resolve()));lib.vgi_stream_reset()
    lib.vgi_stream_token.argtypes=[ctypes.c_long]*3;lib.vgi_stream_token.restype=ctypes.c_long
    lib.vgi_stream_update.argtypes=[ctypes.c_long]*4;lib.vgi_stream_update.restype=ctypes.c_long
    lib.vgi_stream_norm.restype=ctypes.c_double
    lib.vgi_full_gradient.argtypes=[ctypes.c_long];lib.vgi_full_gradient.restype=ctypes.c_double
    lib.vgi_full_parameter.argtypes=[ctypes.c_long];lib.vgi_full_parameter.restype=ctypes.c_double
    native=[];prefix=None
    for frame in frames:
        for index,(token,label) in enumerate(zip(frame['tokens'],frame['labels'])):
            assert lib.vgi_stream_token(index,token,label)==0
        sup,ref=model(torch.tensor([frame['tokens']]))
        kind=frame['kind']
        if kind==0:
            target=torch.tensor(frame['labels']);loss=F.cross_entropy(sup[0],target)
        else:
            width,base=(8,261) if kind==1 else (4,259)
            selected=torch.stack([(ref if k%2 else sup)[0,-1,base+k//2] for k in range(width)])
            target=torch.tensor([(frame['bits']>>k)&1 for k in range(width)],dtype=torch.double)
            loss=F.binary_cross_entropy_with_logits(selected,target)
        optimizer.zero_grad();loss.backward()
        gradients={key:p.grad.detach().clone() for key,p in model.named_parameters()}
        norm=float(torch.sqrt(sum((g*g).sum() for g in gradients.values())))
        clip=build['stream_gradient_clip'];scale=min(1.,clip/norm) if clip>0 and norm>0 else 1.
        for parameter in model.parameters():parameter.grad.mul_(scale)
        optimizer.step()
        observed_loss=lib.vgi_stream_update(kind,frame['bits'],frame['unroll'],frame['steps'])
        if observed_loss<0 or abs(observed_loss/1e6-float(loss.detach()))>1e-6:
            raise RuntimeError(f'loss mismatch at {frame["frame"]}: {observed_loss}')
        if abs(lib.vgi_stream_norm()-norm)>1e-6:raise RuntimeError('gradient norm mismatch')
        index=0;worst_g=worst_p=0.;parameters=dict(model.named_parameters())
        for entry in build['parameters']:
            name,size=entry['key'],entry['size']
            actual_g=torch.tensor([lib.vgi_full_gradient(i) for i in range(index,index+size)],dtype=torch.double)
            actual_p=torch.tensor([lib.vgi_full_parameter(i) for i in range(index,index+size)],dtype=torch.double)
            expected_g=gradients[name].flatten();expected_p=parameters[name].detach().flatten()
            if not torch.allclose(actual_g,expected_g,atol=1e-8,rtol=1e-5):raise RuntimeError('stream gradient mismatch: '+name)
            if not torch.allclose(actual_p,expected_p,atol=1e-8,rtol=1e-5):raise RuntimeError('stream parameter mismatch: '+name)
            worst_g=max(worst_g,float((actual_g-expected_g).abs().max()))
            worst_p=max(worst_p,float((actual_p-expected_p).abs().max()));index+=size
        native.append({'frame':frame['frame'],'task':frame['task'],'loss':float(loss.detach()),
                       'gradient_norm':norm,'gradient_max_abs_error':worst_g,'parameter_max_abs_error':worst_p})
        if prefix is None:prefix={k:v.detach().clone() for k,v in model.state_dict().items()}
        print(json.dumps(native[-1]),flush=True)
    (runtime/'native_stream_verification.json').write_text(json.dumps(native,indent=2)+'\n')

    def execute(payload,name,expected,expected_status):
        command=[str(root/'Vox/target/release/vox'),'run',str(runtime/'model.imasm')]
        env=dict(os.environ);env['VOX_STEPS']=str(max(1000000000,len(frames)*200000000))
        result=subprocess.run(command,input=payload,capture_output=True,timeout=300,env=env)
        (runtime/f'{name}.log').write_bytes(result.stderr)
        if result.returncode!=expected_status:raise RuntimeError(result.stderr.decode(errors='replace'))
        if len(result.stdout)<8:raise RuntimeError('no learned state returned')
        header_len=struct.unpack('<Q',result.stdout[:8])[0];header=json.loads(result.stdout[8:8+header_len])
        end=8+header_len+max(v['data_offsets'][1] for k,v in header.items() if k!='__metadata__')
        if len(result.stdout)!=end:raise RuntimeError('checkpoint return boundary mismatch')
        directory=runtime/name;directory.mkdir(exist_ok=False);path=directory/'model.safetensors';path.write_bytes(result.stdout)
        actual=load_file(str(path));checks=[]
        if set(actual)!=set(expected):raise RuntimeError('tensor set changed')
        for key in actual:
            error=float((actual[key].double()-expected[key].double()).abs().max())
            if not torch.allclose(actual[key].double(),expected[key].double(),atol=1e-8,rtol=1e-5):
                raise RuntimeError('returned state differs: '+key)
            checks.append({'tensor':key,'max_abs_error':error})
        for filename in ('config.json','tokenizer.json'):shutil.copyfile(parent/filename,directory/filename)
        probe=frames[-1] if expected_status==0 else frames[0]
        (directory/'execution_input.json').write_text(json.dumps({'tokens':probe['tokens'],
            'source_id':probe['source_id'],'evidence':'stream checkpoint input context'},indent=2)+'\n')
        return {'command':command,'exit_status':result.returncode,'checkpoint':str(path),
                'checkpoint_sha256':hashlib.sha256(result.stdout).hexdigest(),'tensor_checks':checks,
                'log':result.stderr.decode(errors='replace')}

    print('Executing complete source stream through Vox',flush=True)
    complete=execute(wire,'streamed_checkpoint',model.state_dict(),0)
    expected_lines=len(frames)
    if complete['log'].count('vgi-stream ')!=expected_lines:raise RuntimeError('frame receipt count mismatch')
    print('Checking checkpoint return after a truncated second frame',flush=True)
    truncated=MAGIC+struct.pack('<I',2)+pack(frames[0])+pack(frames[1])[:7]
    held=execute(truncated,'held_prefix_checkpoint',prefix,2)
    if ' -32 ' not in held['log']:raise RuntimeError('truncation frontier missing from receipt')
    report={'status':'passed','native':native,'complete':complete,'held_prefix':held,
            'stream_sha256':hashlib.sha256(wire).hexdigest(),
            'module_sha256':hashlib.sha256((runtime/'model.imasm').read_bytes()).hexdigest(),
            'scope':'successive source-bound language/evidence updates and retained prefix; no per-frame parameter reset'}
    (runtime/'stream_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':'passed','frames':len(frames),'tasks':sorted({f['task'] for f in frames}),
                      'gradient_max_error':max(r['gradient_max_abs_error'] for r in native),
                      'retained_prefix':'passed'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','runtime','stream'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();verify(a.root.resolve(),a.runtime.resolve(),a.stream.resolve())
