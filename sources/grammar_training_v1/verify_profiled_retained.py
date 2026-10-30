#!/usr/bin/env python3
"""Check Vox-profiled whole-source learning against the independent Torch action."""
import argparse, hashlib, json, shutil, sys
from pathlib import Path
import torch
from safetensors.torch import load_file
from retained_conditioning import evaluate
from retained_format import decode_return, stream


def verify(root, runtime, fixture, profile):
    torch.set_num_threads(1)
    sys.path.insert(0, str(root/'bob3ect/reentry_dnn'))
    from dtransformer_torch import DialetheicTransformerTorch, export_to_python
    from dtransformer import Evidence
    status=dict(line.split('=',1) for line in profile.with_suffix('.status').read_text().splitlines())
    assert status['wait_status']=='0' and status['deadline_reached']=='false', status
    build=json.loads((runtime/'build.json').read_text());parent=Path(build['parent'])
    assert Path(status['binary']).resolve()==(runtime/'model').resolve()
    rows=[json.loads(line) for line in (fixture/'records.jsonl').open()]
    payload=(fixture/'train.vgir').read_bytes();assert payload==stream(rows)
    model=DialetheicTransformerTorch(**build['config']).double()
    model.load_state_dict(load_file(str(parent/'model.safetensors')))
    optimizer=torch.optim.SGD(model.parameters(),lr=build['learning_rate'])
    losses=[]
    for row in rows:
        assert row['steps']==1
        optimizer.zero_grad()
        result=evaluate(model,row['prefix'],row['target'],context=build['context_length'],
                        kind=row['kind'],bits=row['bits'])
        for loss in result['loss_windows']:loss.backward()
        for parameter in model.parameters():parameter.grad.div_(result['labels'])
        if build['stream_gradient_clip']>0:
            # Same exact scale as the freestanding runtime, without epsilon.
            norm=torch.sqrt(sum((p.grad*p.grad).sum() for p in model.parameters()))
            scale=min(1.,build['stream_gradient_clip']/float(norm)) if norm>0 else 1.
            for parameter in model.parameters():parameter.grad.mul_(scale)
        optimizer.step();losses.append(result['loss'])
    weights,state=decode_return(profile.with_suffix('.stdout').read_bytes())
    out=runtime/'profiled_returned_checkpoint';out.mkdir(exist_ok=True)
    (out/'model.safetensors').write_bytes(weights)
    actual=load_file(str(out/'model.safetensors'));expected=model.state_dict()
    assert set(actual)==set(expected)
    errors={name:float((actual[name].double()-expected[name].double()).abs().max()) for name in actual}
    assert max(errors.values())<1e-8,errors
    assert state['source_bank']==rows[-1]['prefix']+rows[-1]['target']
    assert state['completed_records']==len(rows) and state['record_complete']==1
    before_error=max(abs(a-float(b)) for k,key in enumerate(('before_settlement_t','before_settlement_f'))
                     for a,b in zip(state[key],result['memory'][k].flatten()))
    assert before_error<1e-6
    settled=export_to_python(model).settle(Evidence(result['memory'][0][0].tolist(),result['memory'][1][0].tolist()))
    assert state['settlement_code']=={'divergent':1,'stable':2,'imaginary':3}[settled.kind]
    settlement_error=max(abs(a-b) for key,values in [('memory_t',settled.stream.t),('memory_f',settled.stream.f)]
                         for a,b in zip(state[key],values[0]))
    assert settlement_error<1e-6
    (out/'retained_state.json').write_text(json.dumps(state,indent=2)+'\n')
    for name in ('config.json','tokenizer.json'):shutil.copyfile(parent/name,out/name)
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    receipt={'status':'passed','execution':'Vox profile-native CPU execution', 'profile_status':status,
             'records':len(rows),'tasks':[r['task'] for r in rows], 'reference_losses':losses,
             'parameter_max_abs_error':max(errors.values()),'parameter_errors':errors,
             'before_settlement_max_abs_error':before_error,'settlement_max_abs_error':settlement_error,
             'settlement':settled.kind,'source_bank_exact':True,
             'input_sha256':hashlib.sha256(payload).hexdigest(),'binary_sha256':sha(runtime/'model'),
             'module_sha256':sha(runtime/'model.imasm'),'checkpoint_sha256':sha(out/'model.safetensors'),
             'profile_stdout_sha256':sha(profile.with_suffix('.stdout'))}
    (runtime/'profiled_retained_verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('status','records','parameter_max_abs_error','settlement','source_bank_exact')}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','runtime','fixture','profile'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();verify(a.root.resolve(),a.runtime.resolve(),a.fixture.resolve(),a.profile.resolve())
