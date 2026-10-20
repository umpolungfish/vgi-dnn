#!/usr/bin/env python3
"""Execute learned graph choices, including errors, through native and Vox paths."""
import argparse
from collections import Counter
import ctypes
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import torch
from safetensors.torch import load_file

from train_distinctions import tokens,ACTIONS
from verify_runtime_learning import words


def verify(root,checkpoint,runtime,tasks):
    torch.set_num_threads(1);sys.path.insert(0,str(root/'bob3ect/reentry_dnn'))
    sys.path.insert(0,str(root/'VGI/sources/GRAMMAR_UNIFIED_RUN'))
    from dtransformer_torch import DialetheicTransformerTorch
    from carrier_transitions import gate,split,fuse
    cfg=json.loads((checkpoint/'config.json').read_text());model=DialetheicTransformerTorch(**cfg).double()
    model.load_state_dict(load_file(str(checkpoint/'model.safetensors')))
    lib=ctypes.CDLL(str((runtime/'model.so').resolve()))
    lib.vgi_policy_act.argtypes=[ctypes.c_long]*3;lib.vgi_policy_act.restype=ctypes.c_long
    lib.vgi_policy_pair.argtypes=[ctypes.c_long]*5;lib.vgi_policy_pair.restype=ctypes.c_long
    rows=[json.loads(l) for l in (tasks/'test.jsonl').open()]
    counts=Counter();observations=[];representatives={}

    @torch.no_grad()
    def prediction(row):
        st,rf=model(torch.tensor([tokens(row)]));s,r=st[0,-1,ACTIONS],rf[0,-1,ACTIONS]
        action=int(s.argmax()) if float(s.max())>0 else -1
        evidence=sum((int(s[i]>0)+2*int(r[i]>0))<<(2*i) for i in range(4))
        return action,evidence,s.tolist(),r.tolist()

    def execute(row,action):
        seed,work=row['seed_bits'],row['work']
        if action==0:return gate(work,seed)
        if action in (1,2):
            arms=list(split(seed,action+1));arms[0]=gate(work,arms[0]);return fuse(arms)
        return seed

    for row in rows:
        pair=words(tokens(row));action,evidence,st,rf=prediction(row);value=execute(row,action)
        for readout,expected in [(0,action),(1,value),(2,evidence)]:
            actual=lib.vgi_policy_act(*pair,readout)
            if actual!=expected:raise RuntimeError(f'native policy mismatch {row["id"]}: {readout} {actual} {expected}')
        valid=action>=0 and bool(row['support'][action]);counts['cases']+=1
        counts['valid_choice']+=valid;counts['feasible']+=any(row['support'])
        counts['held']+=action<0;counts['revised']+=action>=0 and action!=row['previous_action']
        counts['released']+=action==3 and row['previous_action'] in (1,2)
        predicted_both=action>=0 and ((evidence>>(2*action))&3)==3
        counts['chosen_predicted_both']+=predicted_both
        labels=[f'action_{action}']
        if action<0:labels.append('open')
        if valid and predicted_both:labels.append('supported_both')
        if not valid and action>=0:labels.append('incorrect_choice')
        if action==3 and row['previous_action'] in (1,2):labels.append('release')
        for label in labels:representatives.setdefault(label,{'row':row,'action':action,'value':value,'evidence':evidence})

    module=runtime/'model.imasm';vox=root/'Vox/target/release/vox'
    seen_calls={}
    def vm(symbol,args,expected,label):
        identity=(symbol,tuple(args))
        if identity in seen_calls:
            record=seen_calls[identity];assert record['expected']==expected
            record.setdefault('also_checks',[]).append(label);return
        command=[str(vox),'run',symbol,str(module),'--args',','.join(map(str,args))]
        completed=subprocess.run(command,capture_output=True,text=True,timeout=60,check=True)
        output=completed.stdout+completed.stderr;match=re.search(r'\)\s*=\s*(-?\d+)',output)
        if not match or int(match[1])!=expected:raise RuntimeError('VM policy mismatch: '+output)
        record={'label':label,'command':command,'expected':expected,'output':output}
        observations.append(record);seen_calls[identity]=record
        print(json.dumps({'checked':label,'symbol':symbol,'returned':expected}),flush=True)
    for label,example in sorted(representatives.items()):
        row=example['row'];pair=words(tokens(row))
        for readout,name in [(0,'action'),(1,'value'),(2,'evidence')]:
            vm('vgi_policy_act',[*pair,readout],example[name],label+':'+name)
    # Seek an observed branch-to-fused return, retaining the first carrier and
    # all first/second judgments inside one VM call. Selection uses the model.
    found=None
    for first_row in rows:
        a,ea,_,_=prediction(first_row)
        if a not in (1,2) or not first_row['support'][a]:continue
        value=execute(first_row,a)
        second_row={**first_row,'seed_bits':value,'work':'≻','work_index':0,'positive_return':value,
                    'negative_return':value,'previous_action':a,'requested_view':0}
        b,eb,_,_=prediction(second_row)
        if b in (0,3) and execute(second_row,b)==value:
            found=(first_row,second_row,a,b,value,ea,eb);break
    if found is None:raise RuntimeError('no observed branch-to-fused trajectory found')
    first_row,second_row,a,b,value,ea,eb=found
    pair_args=[*words(tokens(first_row)),*words(tokens(second_row))]
    pair_checks=[(0,b),(1,value),(2,a),(3,value),(4,ea|(eb<<8)),
                 (19,value),(23,value),(32,value)]
    for readout,expected in pair_checks:
        native=lib.vgi_policy_pair(*pair_args,readout)
        if native!=expected:raise RuntimeError(f'pair state/trace mismatch {readout}: {native}, {expected}')
        vm('vgi_policy_pair',[*pair_args,readout],expected,'retained_pair')
    report={'status':'passed','counts':dict(counts),'reference_candidates':'independent existing gate/split/fuse functions',
            'selection':'learned support logits; no reference-answer filtering',
            'representatives':representatives,'vox_observations':observations,
            'retained_pair':{'first_query':first_row,'second_query':second_row,'first_action':a,'second_action':b,
                             'carrier_return':value,'first_evidence':ea,'second_evidence':eb},
            'module_sha256':hashlib.sha256(module.read_bytes()).hexdigest(),
            'checkpoint_sha256':hashlib.sha256((checkpoint/'model.safetensors').read_bytes()).hexdigest(),
            'scope':'all held-out query choices/returns checked natively; selected cases and a paired return checked through VM'}
    (runtime/'policy_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'passed','counts':dict(counts),'vm_calls':len(observations),
                      'paired_actions':[a,b],'paired_carrier':value}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','checkpoint','runtime','tasks'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();verify(a.root.resolve(),a.checkpoint.resolve(),a.runtime.resolve(),a.tasks.resolve())
