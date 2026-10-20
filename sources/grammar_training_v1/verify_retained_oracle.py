#!/usr/bin/env python3
"""Check executing oracle retrieval, operation residuals, and learning re-entry."""
import argparse,ctypes,hashlib,json,re,subprocess,sys
from pathlib import Path
import torch
from safetensors.torch import load_file
from distinction_format import tokens,words,ACTIONS
from retained_conditioning import evaluate

def verify(root,runtime,tasks,vm=False):
    torch.set_num_threads(1);sys.path.insert(0,str(root/'bob3ect/reentry_dnn'))
    from dtransformer_torch import DialetheicTransformerTorch
    build=json.loads((runtime/'build.json').read_text());parent=Path(build['parent'])
    initial=load_file(str(parent/'model.safetensors'));cfg=build['config']
    lib=ctypes.CDLL(str(runtime/'model.so'))
    lib.vgi_retained_oracle.argtypes=[ctypes.c_ulong,ctypes.c_ulong,ctypes.c_long,ctypes.c_long]
    lib.vgi_retained_oracle.restype=ctypes.c_long
    for name in ('vgi_retained_oracle_observe','vgi_retained_retrieve'):
        getattr(lib,name).restype=ctypes.c_long
    lib.vgi_retained_retrieve.argtypes=[ctypes.c_long,ctypes.c_long]
    lib.vgi_full_parameter.argtypes=[ctypes.c_long];lib.vgi_full_parameter.restype=ctypes.c_double
    lib.vgi_retained_token.argtypes=[ctypes.c_long,ctypes.c_long]
    lib.vgi_retained_update.argtypes=[ctypes.c_long]*6;lib.vgi_retained_update.restype=ctypes.c_long
    sys.path.insert(0,str(root/'VGI/sources/GRAMMAR_ZETA_SIC_ORACLE'))
    from zeta_oracle import split,fuse
    from fractions import Fraction
    lib.vgi_retained_reset()
    for i in range(cfg['vocab_size']):assert lib.vgi_retained_token(i,i)==0
    assert lib.vgi_retained_update(cfg['vocab_size'],0,1,0,0,1)>=0
    for start in range(0,cfg['vocab_size'],16):
        coordinates=[Fraction(i if i<cfg['vocab_size'] else 0) for i in range(start,start+16)]
        masses=split(coordinates,4);assert fuse(masses,4)==coordinates
        for i in range(start,min(start+16,cfg['vocab_size'])):
            assert lib.vgi_retained_retrieve(i,0)==i
            assert Fraction(lib.vgi_retained_retrieve(i,1),lib.vgi_retained_retrieve(i,2))==masses[i-start]
            assert lib.vgi_retained_retrieve(i,3)==-int(sum(masses)*5)
    assert lib.vgi_retained_retrieve(cfg['vocab_size'],0)==-63
    (runtime/'retrieval_verification.json').write_text(json.dumps({'status':'passed',
        'tokens_checked':cfg['vocab_size'],'reference':'exact Fraction d=4 Gram/dual implementation',
        'signed_dual_term_checked':True,'partial_block_padding':True},indent=2)+'\n')
    rows=[json.loads(l) for l in (tasks/'train.jsonl').open()]
    selected=[rows[i] for i in (0,100,500,1000,2000,3000,5000,7000)]
    checks=[]
    for row in selected:
        model=DialetheicTransformerTorch(**cfg).double();model.load_state_dict(initial)
        bits=sum((int(row['support'][k])<<(2*k))|(int(row['refute'][k])<<(2*k+1)) for k in range(4))
        with torch.no_grad():before=evaluate(model,tokens(row),[],context=16,kind=1,bits=bits)
        pred=sum((int(before['first_logits'][0][261+k]>0)<<(2*k))|
                 (int(before['first_logits'][1][261+k]>0)<<(2*k+1)) for k in range(4))
        steps=8 if pred!=bits else 0
        optimizer=torch.optim.SGD(model.parameters(),lr=build['learning_rate'])
        for _ in range(steps):
            optimizer.zero_grad();result=evaluate(model,tokens(row),[],context=16,kind=1,bits=bits)
            for loss in result['loss_windows']:loss.backward()
            for p in model.parameters():p.grad.div_(result['labels'])
            norm=float(torch.sqrt(sum((p.grad*p.grad).sum() for p in model.parameters())))
            scale=min(1.,build['stream_gradient_clip']/norm) if norm else 1.
            for p in model.parameters():p.grad.mul_(scale)
            optimizer.step()
        lib.vgi_retained_reset();packed=lib.vgi_retained_oracle(*words(tokens(row)),8,30);assert packed>=0
        observed=[lib.vgi_retained_oracle_observe(i) for i in range(13)]
        assert observed[0]==bits and observed[1]==pred and observed[3]==pred^bits and observed[10]==steps
        assert [lib.vgi_retained_oracle_observe(20+k) for k in range(4)]==[c['return_bits'] for c in row['candidates']]
        for i,token in enumerate(tokens(row)):
            assert lib.vgi_retained_retrieve(i,0)==token
            assert lib.vgi_retained_retrieve(i,4)==80*token
        index=0;error=0.;expected=dict(model.named_parameters())
        for entry in build['parameters']:
            actual=torch.tensor([lib.vgi_full_parameter(i) for i in range(index,index+entry['size'])],dtype=torch.double)
            error=max(error,float((actual-expected[entry['key']].detach().flatten()).abs().max()));index+=entry['size']
        assert error<1e-8
        checks.append({'query_id':row['id'],'exact_evidence':bits,'before_residual':observed[3],
                       'after_residual':observed[4],'correction_steps':steps,'before_loss':observed[11]/1e6,
                       'after_loss':observed[12]/1e6,'parameter_max_error':error,'exact_source_retrieval':True})
    receipt={'status':'passed','checks':checks,'query_input':'query fields only; candidate targets derived by executing oracle',
             'retrieval':'exact dimension-four Gram/dual recovery of retained token coordinates'}
    if vm:
        row=selected[-1];lib.vgi_retained_reset();expected=lib.vgi_retained_oracle(*words(tokens(row)),0,30)
        args=','.join(map(str,(*words(tokens(row)),0,30)))
        command=[str(root/'Vox/target/release/vox'),'run','vgi_retained_oracle',str(runtime/'model.imasm'),'--args',args]
        process=subprocess.run(command,capture_output=True,text=True,timeout=180)
        (runtime/'oracle_vm.log').write_text(process.stdout+process.stderr)
        match=re.search(r'\)\s*=\s*(-?\d+)',process.stdout+process.stderr)
        assert process.returncode==0 and match and int(match[1])==expected,(process.returncode,process.stdout,process.stderr)
        receipt['vm']={'command':command,'packed_return':expected,'passed':True}
    receipt['module_sha256']=hashlib.sha256((runtime/'model.imasm').read_bytes()).hexdigest()
    (runtime/'oracle_verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','runtime','tasks'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--vm',action='store_true');a=p.parse_args();verify(a.root.resolve(),a.runtime.resolve(),a.tasks.resolve(),a.vm)
