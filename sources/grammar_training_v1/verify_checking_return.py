#!/usr/bin/env python3
"""Compare carried-context checking, learning, and checkpoint re-entry."""
import argparse,ctypes,hashlib,json,os,re,shutil,subprocess,sys
from pathlib import Path
import torch
from safetensors.torch import load
from checking_format import packet,decode,resume_packet
from build_checking_tasks import words
from export_checking_learning import build as export
from optimizer_state import load_model
from retained_conditioning import evaluate
from torch_reference_optimizer import make_optimizer


def verify(root,runtime,tasks,vm):
    torch.set_num_threads(1);sys.path.insert(0,str(root/'bob3ect/reentry_dnn'))
    from dtransformer_torch import DialetheicTransformerTorch
    config=json.loads((runtime/'build.json').read_text());parent=Path(config['parent'])
    lib=ctypes.CDLL(str(runtime/'model.so'))
    for name,n in [('vgi_retained_token',2),('vgi_retained_update',6),('vgi_check_enter',4),
                   ('vgi_check_continue',1),('vgi_check_train_return',1),('vgi_check_observe',1)]:
        getattr(lib,name).argtypes=[ctypes.c_long]*n;getattr(lib,name).restype=ctypes.c_long
    for name,n in [('vgi_retained_first',2),('vgi_retained_memory',2),('vgi_full_parameter',1),('vgi_optimizer_observe',2)]:
        getattr(lib,name).argtypes=[ctypes.c_long]*n;getattr(lib,name).restype=ctypes.c_double
    lib.vgi_check_stream.restype=ctypes.c_long;lib.vgi_check_return.restype=ctypes.c_long
    lib.vgi_retained_reset()
    rows=[json.loads(l) for l in (tasks/'validation.jsonl').open()];selected={};episode=None
    for row in rows:
        for i,t in enumerate(row['prefix']):assert lib.vgi_retained_token(i,t)==0
        assert lib.vgi_retained_update(12,0,3,0,0,1)>=0
        bits=int(lib.vgi_retained_first(0,265)>0)|(int(lib.vgi_retained_first(1,265)>0)<<1)
        selected.setdefault(bits,row)
        if bits!=row['reference_bits'] and (episode is None or bits==3):episode=row
        if set(selected)=={0,1,2,3} and episode is not None:break
    assert episode is not None
    def save(name):
        path=runtime/name
        with path.open('wb') as handle:
            saved=os.dup(1)
            try:os.dup2(handle.fileno(),1);status=lib.vgi_check_return()
            finally:os.dup2(saved,1);os.close(saved)
        assert status==0;return decode(path.read_bytes())
    def stream(payload,name):
        path=runtime/name;path.write_bytes(payload)
        with path.open('rb') as handle:
            saved=os.dup(0)
            try:os.dup2(handle.fileno(),0);result=lib.vgi_check_stream()
            finally:os.dup2(saved,0);os.close(saved)
        return result
    shapes=[]
    for bits,row in selected.items():
        lib.vgi_retained_reset();assert lib.vgi_check_enter(*words(row['prefix']),3)>=0
        before=save(f'shape_{bits}.stdout');calls=lib.vgi_check_observe(9)
        assert stream(resume_packet(before[2],before[1]),f'shape_{bits}.restore')>=0
        assert lib.vgi_check_observe(9)==calls
        after=save(f'shape_{bits}.restored.stdout')
        assert before==after;shapes.append(bits)
    lib.vgi_retained_reset();model=DialetheicTransformerTorch(**config['config']).double()
    model.load_state_dict(load_model(parent/'model.safetensors'));optimizer=make_optimizer(model,config)
    q=list(episode['prefix']);seed=([.05]*model.d,[.05]*model.d);last_return=None;observations=[]
    reference=episode['checking_reference'];carrier=reference['carrier']
    def exact(query):return int(query[9]-266!=carrier or reference['candidate_refute'])|(int(query[9]-266==carrier and reference['candidate_support'])<<1)
    def reference_forward():
        with torch.no_grad():return evaluate(model,q,[],context=16,kind=3,bits=0,initial_memory=seed)
    def observe(result):
        nonlocal last_return
        predicted=int(result['first_logits'][0][265]>0)|(int(result['first_logits'][1][265]>0)<<1)
        last_return=carrier if predicted&1 else q[9]-266
        trace=[lib.vgi_check_observe(i) for i in range(8)]
        assert trace[0]==predicted and trace[3]==last_return and trace[5]==predicted&1
        memory=tuple(v.flatten().tolist() for v in result['memory'])
        error=max(abs(lib.vgi_retained_memory(j,i)-memory[j][i]) for j in range(2) for i in range(model.d))
        assert error<1e-8,error
        observations.append({'predicted':predicted,'returned':last_return,'initial_t':list(seed[0]),'memory_max_error':error})
        return memory
    def train():
        target=exact(q);optimizer.zero_grad();result=evaluate(model,q,[],context=16,kind=3,bits=target,initial_memory=seed)
        predicted=int(result['first_logits'][0][265]>0)|(int(result['first_logits'][1][265]>0)<<1)
        if predicted!=target:
            for loss in result['loss_windows']:loss.backward()
            for p in model.parameters():p.grad.div_(result['labels'])
            norm=float(torch.sqrt(sum((p.grad*p.grad).sum() for p in model.parameters())))
            scale=min(1.,config['stream_gradient_clip']/norm) if norm else 1.
            for p in model.parameters():p.grad.mul_(scale)
            optimizer.step()
        assert lib.vgi_check_train_return(1)>=0
        result=reference_forward();memory=tuple(v.flatten().tolist() for v in result['memory'])
        error=max(abs(lib.vgi_retained_memory(j,i)-memory[j][i]) for j in range(2) for i in range(model.d));assert error<1e-8,error
        return memory
    assert lib.vgi_check_enter(*words(q),3)>=0;observe(reference_forward())
    initial=save('initial_episode.stdout');train();partial=save('partial_episode.stdout')
    assert partial[2]['phase']==2 and partial[2]['teacher_trace'][10]==1
    for _ in range(2):
        next_seed=train();q[9]=266+last_return;q[6]=q[8];seed=next_seed
        assert lib.vgi_check_continue(3)>=0;observe(reference_forward())
    expected=save('continued_episode.stdout')
    parameters=dict(model.named_parameters());position=0;parameter_error=0.;moment_error=0.
    for entry in config['parameters']:
        p=parameters[entry['key']];n=entry['size']
        actual=torch.tensor([lib.vgi_full_parameter(i) for i in range(position,position+n)],dtype=torch.double)
        parameter_error=max(parameter_error,float((actual-p.detach().flatten()).abs().max()))
        for kind,key in [(1,'exp_avg'),(2,'exp_avg_sq')]:
            actual=torch.tensor([lib.vgi_optimizer_observe(kind,i) for i in range(position,position+n)],dtype=torch.double)
            moment_error=max(moment_error,float((actual-optimizer.state[p][key].flatten()).abs().max()))
        position+=n
    assert max(parameter_error,moment_error)<1e-8
    controls=[]
    good=resume_packet(expected[2],expected[1])
    malformed=[]
    changed=bytearray(good);changed[40+8+9*8]^=1;malformed.append(('checksum',bytes(changed)))
    changed=bytearray(good);changed[40+280+16*model.d+128]^=1;malformed.append(('retained_memory',bytes(changed)))
    malformed.append(('trailing',good+b'x'));malformed.append(('wrong_model',resume_packet(initial[2],initial[1])))
    bad=list(q);bad[10]=1;malformed.append(('invalid_query',packet(bad)))
    before=(runtime/'continued_episode.stdout').read_bytes()
    for name,payload in malformed:
        calls=lib.vgi_check_observe(9);code=stream(payload,f'bad_{name}.input');assert code<0
        save('control_after.stdout');assert (runtime/'control_after.stdout').read_bytes()==before
        assert lib.vgi_check_observe(9)==calls;controls.append({'case':name,'status':code,'previous_return_exact':True})
    carried_logits=[lib.vgi_retained_first(i,265) for i in (0,1)]
    assert lib.vgi_retained_update(12,0,3,0,0,1)>=0
    neutral_logits=[lib.vgi_retained_first(i,265) for i in (0,1)]
    context_effect=max(abs(a-b) for a,b in zip(carried_logits,neutral_logits));assert context_effect>1e-10
    checkpoint=runtime/'partial_checkpoint';checkpoint.mkdir(exist_ok=True);(checkpoint/'model.safetensors').write_bytes(partial[0])
    for name in ('config.json','tokenizer.json','execution_input.json'):shutil.copy2(parent/name,checkpoint/name)
    resumed=runtime/'resumed_runtime';export(root,checkpoint,resumed,config['optimizer'])
    restore=resume_packet(partial[2],partial[1]);(resumed/'restore.input').write_bytes(restore)
    result=subprocess.run([str(resumed/'model')],input=restore,capture_output=True,timeout=90);assert result.returncode==0,result.stderr
    restored=decode(result.stdout);assert restored[1:]==partial[1:]
    assert all(torch.equal(load(restored[0])[k],load(partial[0])[k]) for k in load(partial[0]))
    request=resume_packet(partial[2],partial[1],rounds=2,steps=1);(resumed/'continue.input').write_bytes(request)
    result=subprocess.run([str(resumed/'model')],input=request,capture_output=True,timeout=90);assert result.returncode==0,result.stderr
    (resumed/'continued.stdout').write_bytes(result.stdout);actual=decode(result.stdout)
    assert actual[1:]==expected[1:]
    a,b=load(actual[0]),load(expected[0]);assert all(torch.equal(a[k],b[k]) for k in a)
    report={'status':'passed','shape_restores':sorted(shapes),'context_logit_effect':context_effect,
            'parameter_max_error':parameter_error,'moment_max_error':moment_error,'observations':observations,
            'reloaded_parameter_and_optimizer_tensors_exact':True,'reloaded_retained_memory_and_checking_state_exact':True,
            'restore_added_oracle_calls':0,'admission_controls':controls,'source_example_id':episode['example_id'],
            'source_sha256':hashlib.sha256((tasks/'validation.jsonl').read_bytes()).hexdigest()}
    if vm:
        from checking_format import resume_packet as resume
        calls_request=resume(partial[2],partial[1],readout=18)
        command=[str(root/'Vox/target/release/vox'),'run','vgi_check_stream',str(resumed/'model.imasm')]
        result=subprocess.run(command,input=calls_request,capture_output=True,timeout=180)
        text=(result.stdout+result.stderr).decode();(resumed/'restore_vm.log').write_text(text)
        match=re.search(r'\)\s*=\s*(-?\d+)',text);assert result.returncode==0 and match and int(match[1])==0,text
        report['vm_restore_oracle_calls']=0
    (runtime/'checking_return_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','runtime','tasks'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--vm',action='store_true');a=p.parse_args();verify(a.root.resolve(),a.runtime.resolve(),a.tasks.resolve(),a.vm)
