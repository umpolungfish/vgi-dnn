#!/usr/bin/env python3
"""Compare sampled Vox-returned policy updates with independent full-gradient steps."""
import argparse,hashlib,json,sys
from pathlib import Path
import torch
from safetensors.torch import load,load_file
from retained_format import decode_capsules
from retained_conditioning import evaluate
from optimizer_state import split_state
from torch_reference_optimizer import make_optimizer

def verify(root,runtime,batch,profile):
    torch.set_num_threads(1);sys.path.insert(0,str(root/'bob3ect/reentry_dnn'))
    from dtransformer_torch import DialetheicTransformerTorch
    build=json.loads((runtime/'build.json').read_text())
    rows=[json.loads(l) for l in (batch/'records.jsonl').open()]
    capsules,tail=decode_capsules(profile.with_suffix('.stdout').read_bytes());assert not tail
    snapshots={state['completed_records']:weights for weights,state in capsules}
    indices=[i for i,r in enumerate(rows) if r['kind'] in (1,3,4,5)]
    def target_bits(row):
        if row['kind']==1:return row['bits']
        if row['kind'] in (4,5):return row['reference_bits']
        reference=row['reference']
        return sum((int(reference['support'][k])<<(2*k))|(int(reference['refute'][k])<<(2*k+1)) for k in range(4))
    chosen=sorted({indices[0],indices[-1],*[indices[len(indices)*k//8] for k in range(1,8)]})
    chosen=sorted(set(chosen)|{next(i for i in indices if target_bits(rows[i])==bits)
                              for bits in {target_bits(rows[i]) for i in indices}})
    checks=[]
    for index in chosen:
        before,optimizer_state=split_state(load(snapshots[index]) if index else load_file(str(Path(build['parent'])/'model.safetensors')))
        model=DialetheicTransformerTorch(**build['config']).double();model.load_state_dict(before)
        row=rows[index];bits=target_bits(row)
        supervision=3 if row['kind']==5 else 2 if row['kind']==4 else 1
        base,width=(265,2) if supervision==3 else (259,4) if supervision==2 else (261,8)
        with torch.no_grad():probe=evaluate(model,row['prefix'],[],context=build['context_length'],kind=supervision,bits=bits)
        prediction=sum(int(probe['first_logits'][k%2][base+k//2]>0)<<k for k in range(width))
        steps=row['steps'] if row['kind']==1 or prediction!=bits else 0
        optimizer=make_optimizer(model,build,optimizer_state)
        for _ in range(steps):
            optimizer.zero_grad();result=evaluate(model,row['prefix'],[],context=build['context_length'],kind=supervision,bits=bits)
            for loss in result['loss_windows']:loss.backward()
            for p in model.parameters():p.grad.div_(result['labels'])
            norm=float(torch.sqrt(sum((p.grad*p.grad).sum() for p in model.parameters())))
            scale=min(1.,build['stream_gradient_clip']/norm) if norm>0 else 1.
            for p in model.parameters():p.grad.mul_(scale)
            optimizer.step()
        actual,actual_optimizer=split_state(load(snapshots[index+1]));expected=model.state_dict()
        error=max(float((actual[k].double()-expected[k].double()).abs().max()) for k in actual)
        assert error<1e-8,error
        moment_error=0.
        if actual_optimizer:
            clock=int(optimizer_state['clock'][0]) if optimizer_state else 0
            assert int(actual_optimizer['clock'][0])==clock+steps
            position=0;parameters=dict(model.named_parameters())
            for entry in build['parameters']:
                p=parameters[entry['key']];end=position+entry['size'];state=optimizer.state.get(p,{})
                for name,key in [('m','exp_avg'),('v','exp_avg_sq')]:
                    expected_moment=state.get(key,torch.zeros_like(p)).flatten()
                    moment_error=max(moment_error,float((actual_optimizer[name][position:end]-expected_moment).abs().max()))
                position=end
            assert moment_error<1e-8,moment_error
        checks.append({'record':index+1,'example_id':row['example_id'],'bits':bits,'reference_steps':steps,
                       'reference_loss':probe['loss'],'parameter_max_abs_error':error,'moment_max_abs_error':moment_error})
    report={'status':'passed','scope':'sampled whole-network candidate-evidence updates on actual returned trajectory',
            'checks':checks,'max_parameter_error':max(c['parameter_max_abs_error'] for c in checks),
            'training_evidence_states':sorted({(target_bits(r)>>(2*k))&3 for r in rows if r['kind'] in (1,3,4,5) for k in range(1 if r['kind']==5 else 2 if r['kind']==4 else 4)}),
            'input_sha256':hashlib.sha256((batch/'train.vgir').read_bytes()).hexdigest()}
    (runtime/'policy_update_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','runtime','batch','profile'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();verify(a.root.resolve(),a.runtime.resolve(),a.batch.resolve(),a.profile.resolve())
