#!/usr/bin/env python3
"""Verify exact source retrieval, full identities, provenance and learning action."""
import argparse,ctypes,hashlib,json,re,subprocess,sys
from pathlib import Path
import torch
from safetensors.torch import load_file
from retained_conditioning import evaluate
from optimizer_state import load_model
from torch_reference_optimizer import make_optimizer


def verify(root,runtime,records,vm=False):
    torch.set_num_threads(1)
    rows=[json.loads(line) for line in records.open()];build=json.loads((runtime/'build.json').read_text())
    assert hashlib.sha256(records.read_bytes()).hexdigest()==build['corpus_oracle']['source_sha256']
    lib=ctypes.CDLL(str(runtime/'model.so'))
    lib.vgi_corpus_find.argtypes=[ctypes.c_ulong]*4;lib.vgi_corpus_find.restype=ctypes.c_long
    for name,args in [('vgi_corpus_metadata',2),('vgi_corpus_retrieve',3),('vgi_corpus_select',1),('vgi_corpus_learn',3),('vgi_retained_retrieve',2)]:
        getattr(lib,name).argtypes=[ctypes.c_long]*args;getattr(lib,name).restype=ctypes.c_long
    lib.vgi_full_parameter.argtypes=[ctypes.c_long];lib.vgi_full_parameter.restype=ctypes.c_double
    checks=[]
    for index,row in enumerate(rows):
        key=hashlib.sha256(row['example_id'].encode()).digest();words=[int.from_bytes(key[i:i+8],'little') for i in range(0,32,8)]
        assert lib.vgi_corpus_find(*words)==index
        bad=words.copy();bad[-1]^=1;assert lib.vgi_corpus_find(*bad)==-70
        size=lib.vgi_corpus_metadata(index,-1)
        metadata=json.loads(bytes(lib.vgi_corpus_metadata(index,i) for i in range(size)))
        assert {k:v for k,v in metadata.items() if k not in ('lookup_sha256','token_bank_sha256')}=={k:v for k,v in row.items() if k not in ('prefix','target')}
        values=row['prefix']+row['target'];assert lib.vgi_corpus_retrieve(index,-1,0)==len(values)
        recovered=[lib.vgi_corpus_retrieve(index,i,0) for i in range(len(values))];assert recovered==values
        digest=hashlib.sha256(b''.join(v.to_bytes(2,'little') for v in recovered)).hexdigest()
        assert digest==metadata['token_bank_sha256'] and metadata['lookup_sha256']==key.hex()
        assert lib.vgi_corpus_select(index)==len(values)
        assert all(lib.vgi_retained_retrieve(i,0)==v for i,v in enumerate(values))
        assert lib.vgi_corpus_retrieve(index,len(values),0)==-71
        checks.append({'record':index,'example_id':row['example_id'],'tokens':len(values),'task':row['task'],
                       'full_identity_lookup':True,'metadata_exact':True,'source_bank_exact':True,'token_sha256':digest})
    # An independently differentiated complete-source operation checks that retrieved
    # coordinates actually supply the training input and target.
    sys.path.insert(0,str(root/'bob3ect/reentry_dnn'))
    from dtransformer_torch import DialetheicTransformerTorch
    parent=Path(build['parent']);model=DialetheicTransformerTorch(**build['config']).double()
    model.load_state_dict(load_model(parent/'model.safetensors'))
    artifact_indices=[i for i,row in enumerate(rows) if row['task'].startswith('ob3ect_artifact_')]
    learning_index=min(artifact_indices,key=lambda i:len(rows[i]['prefix'])+len(rows[i]['target'])) if artifact_indices else 0
    row=rows[learning_index];optimizer=make_optimizer(model,build);optimizer.zero_grad()
    result=evaluate(model,row['prefix'],row['target'],context=build['context_length'],kind=row['kind'],bits=row['bits'])
    for loss in result['loss_windows']:loss.backward()
    for p in model.parameters():p.grad.div_(result['labels'])
    norm=float(torch.sqrt(sum((p.grad*p.grad).sum() for p in model.parameters())))
    scale=min(1.,build['stream_gradient_clip']/norm) if norm else 1.
    for p in model.parameters():p.grad.mul_(scale)
    optimizer.step();lib.vgi_retained_reset();observed=lib.vgi_corpus_learn(learning_index,1,0)
    assert abs(observed/1e6-result['loss'])<1e-6
    expected=dict(model.named_parameters());position=0;error=0.
    for entry in build['parameters']:
        actual=torch.tensor([lib.vgi_full_parameter(i) for i in range(position,position+entry['size'])],dtype=torch.double)
        error=max(error,float((actual-expected[entry['key']].detach().flatten()).abs().max()));position+=entry['size']
    assert error<1e-8
    optimizer_errors={}
    if build.get('optimizer')=='adam':
        lib.vgi_optimizer_observe.argtypes=[ctypes.c_long]*2;lib.vgi_optimizer_observe.restype=ctypes.c_double
        position=0
        for kind,key in ((1,'exp_avg'),(2,'exp_avg_sq')):
            worst=0.;position=0
            for entry in build['parameters']:
                p=expected[entry['key']];size=entry['size']
                actual=torch.tensor([lib.vgi_optimizer_observe(kind,i) for i in range(position,position+size)],dtype=torch.double)
                worst=max(worst,float((actual-optimizer.state[p][key].flatten()).abs().max()));position+=size
            optimizer_errors[key]=worst
        assert max(optimizer_errors.values())<1e-8
        assert lib.vgi_optimizer_observe(0,0)==build['optimizer_initial_clock']+1
    report={'status':'passed','records':checks,'tokens_checked':sum(r['tokens'] for r in checks),
            'independent_retrieved_learning_parameter_error':error,'source_sha256':hashlib.sha256(records.read_bytes()).hexdigest(),
            'learning_record':learning_index,'learning_task':row['task'],'optimizer':build.get('optimizer','sgd'),
            'optimizer_moment_errors':optimizer_errors}
    (runtime/'corpus_oracle_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    if vm:
        index=learning_index if artifact_indices else len(rows)-1
        values=rows[index]['prefix']+rows[index]['target'];offset=len(values)//2;expected=values[offset]
        command=[str(root/'Vox/target/release/vox'),'run','vgi_corpus_retrieve',str(runtime/'model.imasm'),'--args',f'{index},{offset},0']
        process=subprocess.run(command,capture_output=True,text=True,timeout=90)
        observation=process.stdout+process.stderr;(runtime/'corpus_oracle_vm.log').write_text(observation)
        match=re.search(r'\)\s*=\s*(-?\d+)',observation)
        assert process.returncode==0 and match and int(match[1])==expected,observation
        report['vm']={'command':command,'returned_token':expected,'passed':True}
        (runtime/'corpus_oracle_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'records':len(checks),'tokens_checked':report['tokens_checked'],
                      'parameter_error':error,'vm':report.get('vm')}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','runtime','records'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--vm',action='store_true');a=p.parse_args();verify(a.root.resolve(),a.runtime.resolve(),a.records.resolve(),a.vm)
