#!/usr/bin/env python3
"""Verify whole-source credit assignment, checkpoint boundaries, and state return."""
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
import time

import torch
from safetensors.torch import load_file

from retained_conditioning import evaluate
from retained_format import MAGIC,pack,stream,decode_return


def verify(root,runtime,fixture,vm_all=False,vm_window=None):
    torch.set_num_threads(1);sys.path.insert(0,str(root/'bob3ect/reentry_dnn'))
    from dtransformer_torch import DialetheicTransformerTorch,export_to_python
    from dtransformer import Evidence
    build=json.loads((runtime/'build.json').read_text());parent=Path(build['parent']);cfg=build['config']
    initial=load_file(str(parent/'model.safetensors'));context=build['context_length']
    rows=[json.loads(l) for l in (fixture/'records.jsonl').open()]
    assert (fixture/'train.vgir').read_bytes()==stream(rows)
    lib=ctypes.CDLL(str((runtime/'model.so').resolve()))
    lib.vgi_retained_token.argtypes=[ctypes.c_long]*2;lib.vgi_retained_token.restype=ctypes.c_long
    lib.vgi_retained_update.argtypes=[ctypes.c_long]*6;lib.vgi_retained_update.restype=ctypes.c_long
    for name in ('vgi_full_parameter','vgi_full_gradient'):
        getattr(lib,name).argtypes=[ctypes.c_long];getattr(lib,name).restype=ctypes.c_double
    for name in ('vgi_retained_memory','vgi_retained_first'):
        getattr(lib,name).argtypes=[ctypes.c_long]*2;getattr(lib,name).restype=ctypes.c_double
    lib.vgi_retained_nodes.restype=ctypes.c_ulong

    def new_model():
        m=DialetheicTransformerTorch(**cfg).double();m.load_state_dict(initial);return m

    def reference_step(model,row):
        optimizer=torch.optim.SGD(model.parameters(),lr=build['learning_rate']);optimizer.zero_grad()
        result=evaluate(model,row['prefix'],row['target'],context=context,bptt=0,kind=row['kind'],bits=row['bits'])
        for loss in result['loss_windows']:loss.backward()
        for p in model.parameters():p.grad.div_(result['labels'])
        gradients={name:p.grad.detach().clone() for name,p in model.named_parameters()}
        norm=float(torch.sqrt(sum((g*g).sum() for g in gradients.values())))
        clip=build['stream_gradient_clip'];scale=min(1.,clip/norm) if clip>0 and norm>0 else 1.
        for p in model.parameters():p.grad.mul_(scale)
        optimizer.step();return result,gradients,norm

    def native(row,steps=None,window=None):
        for i,token in enumerate(row['prefix']+row['target']):assert lib.vgi_retained_token(i,token)==0
        return lib.vgi_retained_update(len(row['prefix']),len(row['target']),row['kind'],row['bits'],
                    row['steps'] if steps is None else steps,row['checkpoint_chunks'] if window is None else window)

    def compare_parameters(model,gradients=None):
        parameters=dict(model.named_parameters());index=0;worst_p=worst_g=0.
        for entry in build['parameters']:
            name,size=entry['key'],entry['size']
            actual=torch.tensor([lib.vgi_full_parameter(i) for i in range(index,index+size)],dtype=torch.double)
            expected=parameters[name].detach().flatten()
            if not torch.allclose(actual,expected,atol=1e-8,rtol=1e-5):raise RuntimeError('retained parameter mismatch: '+name)
            worst_p=max(worst_p,float((actual-expected).abs().max()))
            if gradients is not None:
                actual_g=torch.tensor([lib.vgi_full_gradient(i) for i in range(index,index+size)],dtype=torch.double)
                expected_g=gradients[name].flatten()
                if not torch.allclose(actual_g,expected_g,atol=1e-8,rtol=1e-5):raise RuntimeError('whole-source gradient mismatch: '+name)
                worst_g=max(worst_g,float((actual_g-expected_g).abs().max()))
            index+=size
        return worst_p,worst_g

    lib.vgi_retained_reset();model=new_model();native_checks=[]
    for row in rows:
        reference,gradients,norm=reference_step(model,row)
        observed=native(row)
        if observed<0 or abs(observed/1e6-reference['loss'])>1e-6:raise RuntimeError(f'retained loss mismatch: {observed}')
        pe,ge=compare_parameters(model,gradients)
        memory_error=max(abs(getattr(lib,'vgi_retained_memory')(channel,c)-float(reference['memory'][channel][0,0,c]))
                         for channel in range(2) for c in range(cfg['d_model']))
        if memory_error>1e-6:raise RuntimeError('retained carrier mismatch')
        check={'task':row['task'],'source_id':row['source_id'],'prefix_tokens':len(row['prefix']),
               'target_tokens':len(row['target']),'source_tokens_processed':reference['source_tokens_processed'],
               'loss':reference['loss'],'gradient_max_abs_error':ge,'parameter_max_abs_error':pe,
               'memory_max_abs_error':memory_error,'peak_tape_nodes':lib.vgi_retained_nodes()}
        native_checks.append(check);print(json.dumps(check),flush=True)
    (runtime/'native_retained_verification.json').write_text(json.dumps(native_checks,indent=2)+'\n')

    # Identical suffix, changed earliest lexical token. Check forward sensitivity
    # and full credit assignment under different recomputation partitions.
    controls=[];firsts=[];updates=[]
    offset=sum(e['size'] for e in build['parameters'][:next(i for i,e in enumerate(build['parameters']) if e['key']=='E_t')])
    for first,window in [(65,0),(65,1),(65,2),(65,4),(66,4)]:
        row={'prefix':[256,first]+[120]*70+[257],'target':[97,258],
             'kind':0,'bits':0,'steps':1,'checkpoint_chunks':window}
        lib.vgi_retained_reset();m=new_model();ref,grads,norm=reference_step(m,row)
        observed=native(row);assert observed>=0 and abs(observed/1e6-ref['loss'])<1e-6
        pe,ge=compare_parameters(m,grads)
        early=torch.tensor([lib.vgi_full_gradient(offset+first*cfg['d_model']+c) for c in range(cfg['d_model'])])
        if float(early.norm())==0:raise RuntimeError('early source lost gradient credit')
        logits=[lib.vgi_retained_first(0,c) for c in range(cfg['vocab_size'])]
        firsts.append(logits);updates.append([lib.vgi_full_parameter(i) for i in range(build['parameter_count'])])
        controls.append({'early_token':first,'checkpoint_chunks':window,'loss':ref['loss'],
                         'early_embedding_gradient_norm':float(early.norm()),'gradient_error':ge,'parameter_error':pe})
    partition_error=max(abs(a-b) for v in updates[1:-1] for a,b in zip(updates[0],v))
    sensitivity=max(abs(a-b) for a,b in zip(firsts[-2],firsts[-1]))
    if partition_error>1e-8 or sensitivity<1e-9:raise RuntimeError('partition invariance or source sensitivity failed')
    (runtime/'conditioning_controls.json').write_text(json.dumps({'cases':controls,
        'checkpoint_partition_update_error':partition_error,'same_suffix_early_prefix_logit_difference':sensitivity},indent=2)+'\n')
    print(json.dumps({'conditioning_controls':'passed','partition_error':partition_error,'early_prefix_effect':sensitivity}),flush=True)

    selected=rows if vm_all else [r for r in rows if r['task'] in ('catalog_to_tuple','ob3ect_recorded_construction','candidate_evidence')]
    if vm_window is not None:selected=[{**r,'checkpoint_chunks':vm_window} for r in selected]
    reference_model=new_model();prefix_model=None;last_result=None
    for row in selected:
        last_result,_,_=reference_step(reference_model,row)
        if prefix_model is None:prefix_model={k:v.detach().clone() for k,v in reference_model.state_dict().items()}

    def execute(payload,label,expected_state,expected_row,expected_result,exit_status,completed):
        command=[str(root/'Vox/target/release/vox'),'run',str(runtime/'model.imasm')]
        env=dict(os.environ);env['VOX_STEPS']='50000000000'
        started=time.monotonic()
        (runtime/f'{label}.input.vgir').write_bytes(payload)
        with (runtime/f'{label}.log').open('wb') as log:
            try:run=subprocess.run(command,input=payload,stdout=subprocess.PIPE,stderr=log,timeout=600,env=env)
            except subprocess.TimeoutExpired as error:
                (runtime/f'{label}.partial.stdout').write_bytes(error.output or b'')
                (runtime/f'{label}.timeout.json').write_text(json.dumps({'timeout_seconds':600,
                    'command':command,'complete_state_verified':False,'stdout_bytes_observed':len(error.output or b''),
                    'input_sha256':hashlib.sha256(payload).hexdigest()},indent=2)+'\n')
                raise
        error_log=(runtime/f'{label}.log').read_bytes()
        elapsed=time.monotonic()-started
        if run.returncode!=exit_status:raise RuntimeError(error_log.decode(errors='replace'))
        weights,returned=decode_return(run.stdout);out=runtime/label;out.mkdir(exist_ok=False)
        (out/'model.safetensors').write_bytes(weights);(out/'retained_state.json').write_text(json.dumps(returned,indent=2)+'\n')
        actual=load_file(str(out/'model.safetensors'))
        for name in actual:
            if not torch.allclose(actual[name].double(),expected_state[name].double(),atol=1e-8,rtol=1e-5):
                raise RuntimeError('VM retained checkpoint mismatch: '+name)
        if returned['source_bank']!=expected_row['prefix']+expected_row['target']:raise RuntimeError('source bank was not retained exactly')
        if returned['completed_records']!=completed or returned['record_complete']!=1:raise RuntimeError('continuation cursor mismatch')
        expected_memory=expected_result['memory']
        for channel,key in enumerate(('before_settlement_t','before_settlement_f')):
            if max(abs(a-float(b)) for a,b in zip(returned[key],expected_memory[channel].flatten()))>1e-6:
                raise RuntimeError('pre-return memory mismatch')
        if build.get('retained_settlement') and exit_status==0:
            final_model=new_model();final_model.load_state_dict(expected_state);py=export_to_python(final_model)
            settlement=py.settle(Evidence(expected_memory[0][0].tolist(),expected_memory[1][0].tolist()))
            if returned['settlement_code']!={'divergent':1,'stable':2,'imaginary':3}[settlement.kind]:
                raise RuntimeError('retained settlement mismatch')
            for key,matrix in [('memory_t',settlement.stream.t),('memory_f',settlement.stream.f)]:
                if max(abs(a-b) for a,b in zip(returned[key],matrix[0]))>1e-6:raise RuntimeError('settled carrier mismatch')
        for filename in ('config.json','tokenizer.json'):shutil.copyfile(parent/filename,out/filename)
        receipt={'command':command,'seconds':elapsed,'completed_records':completed,'state':str(out/'retained_state.json'),
                 'source_bank_exact':True,'checkpoint_sha256':hashlib.sha256(weights).hexdigest(),
                 'exit_status':run.returncode,'settlement_code':returned['settlement_code'],'log':error_log.decode(errors='replace')}
        return receipt

    print('Executing complete retained catalog, ob3ect, and evidence records through Vox',flush=True)
    complete=execute(stream(selected),'returned_checkpoint',reference_model.state_dict(),selected[-1],last_result,0,len(selected))
    prefix_ref=new_model();prefix_result,_,_=reference_step(prefix_ref,selected[0])
    truncated=MAGIC+struct.pack('<I',2)+pack(selected[0])+pack(selected[1])[:9]
    print('Checking retained source and parameter return at a partial-record boundary',flush=True)
    held=execute(truncated,'held_checkpoint',prefix_ref.state_dict(),selected[0],prefix_result,2,1)
    report={'status':'passed','native':native_checks,'controls':controls,
            'partition_update_error':partition_error,'early_prefix_logit_effect':sensitivity,
            'vm_complete':complete,'vm_held':held,'vm_tasks':[r['task'] for r in selected],
            'scope':'whole-source conditioning, full gradients, exact source-bank return, checkpointed re-entry',
            'module_sha256':hashlib.sha256((runtime/'model.imasm').read_bytes()).hexdigest()}
    (runtime/'retained_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':'passed','native_records':len(rows),'vm_records':len(selected),
                      'early_prefix_effect':sensitivity,'partition_update_error':partition_error}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','runtime','fixture'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--vm-all',action='store_true');p.add_argument('--vm-window',type=int)
    a=p.parse_args();verify(a.root.resolve(),a.runtime.resolve(),a.fixture.resolve(),a.vm_all,a.vm_window)
