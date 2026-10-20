#!/usr/bin/env python3
"""Measure learned checking and trace actual conditional source-oracle calls."""
import argparse,ctypes,hashlib,json,re,struct,subprocess,sys
from collections import Counter,defaultdict
from pathlib import Path
import torch
from optimizer_state import load_model
from retained_conditioning import evaluate
from build_checking_tasks import words


def verify(root,runtime,tasks,split,vm):
    torch.set_num_threads(1);sys.path.insert(0,str(root/'bob3ect/reentry_dnn'))
    from dtransformer_torch import DialetheicTransformerTorch
    build=json.loads((runtime/'build.json').read_text());parent=Path(build['parent'])
    model=DialetheicTransformerTorch(**build['config']).double();model.load_state_dict(load_model(parent/'model.safetensors'))
    lib=ctypes.CDLL(str(runtime/'model.so'));lib.vgi_retained_reset()
    lib.vgi_check_execute.argtypes=[ctypes.c_ulong]*3+[ctypes.c_long];lib.vgi_check_execute.restype=ctypes.c_long
    lib.vgi_check_observe.argtypes=[ctypes.c_long];lib.vgi_check_observe.restype=ctypes.c_long
    lib.vgi_full_parameter.argtypes=[ctypes.c_long];lib.vgi_full_parameter.restype=ctypes.c_double
    lib.vgi_optimizer_observe.argtypes=[ctypes.c_long]*2;lib.vgi_optimizer_observe.restype=ctypes.c_double
    parameters=lambda:[lib.vgi_full_parameter(i) for i in range(build['parameter_count'])]
    initial=parameters();initial_clock=lib.vgi_optimizer_observe(0,0)
    source=tasks/f'{split}.jsonl';rows=[json.loads(l) for l in source.open()]
    counts=Counter();groups=defaultdict(Counter);observed={};predictions=[]
    for index,row in enumerate(rows):
        q=row['prefix'];a,b,c=words(q);before=lib.vgi_check_observe(8)
        packed=lib.vgi_check_execute(a,b,c,3);assert packed>=0
        trace=[lib.vgi_check_observe(i) for i in range(8)];after=lib.vgi_check_observe(8)
        bits,asked,proposal,returned,action,calls,reference,held=trace
        expected=row['reference_bits'];carrier=row['checking_reference']['carrier']
        assert bits==packed&3 and asked==bits&1 and calls==asked==after-before
        assert proposal==q[9]-266 and action==q[8]-290 and held==int(bits in (0,3))
        assert returned==(carrier if asked else proposal)
        assert reference==(expected if asked else -1)
        if index<8:
            with torch.no_grad():result=evaluate(model,q,[],context=16,kind=3,bits=0)
            independent=int(result['first_logits'][0][265]>0)|(int(result['first_logits'][1][265]>0)<<1)
            assert bits==independent
        group=groups[expected]
        for counter in (counts,group):
            counter['queries']+=1;counter['evidence_exact']+=bits==expected;counter['check_bit_exact']+=(bits&1)==(expected&1)
            counter['oracle_calls']+=calls;counter['carrier_exact']+=returned==carrier
            counter['unchecked_wrong_proposals']+=not asked and proposal!=carrier
            counter['held']+=held
        predictions.append({'id':row['example_id'],'expected':expected,'prediction':bits,'proposal':proposal,
                            'returned':returned,'source_carrier':carrier,'oracle_calls':calls,'held':held})
        priority=2*int(proposal!=carrier)+int(carrier!=0)
        if bits not in observed or priority>observed[bits][2]:observed[bits]=(q,packed,priority)
    assert parameters()==initial and lib.vgi_optimizer_observe(0,0)==initial_clock
    bad=list(rows[0]['prefix']);bad[10]=1;a,b,c=words(bad);before=lib.vgi_check_observe(8)
    assert lib.vgi_check_execute(a,b,c,3)==-100 and lib.vgi_check_observe(8)==before
    report={'status':'passed','split':split,'counts':dict(counts),'per_reference_shape':dict(groups),
            'observed_prediction_shapes':sorted(observed),'macro_evidence_accuracy':sum(g['evidence_exact']/g['queries'] for g in groups.values())/len(groups),
            'forward_only_parameters_exact':True,'optimizer_clock_unchanged':True,'invalid_query_rejected_before_oracle':True,
            'independent_torch_proposals':min(8,len(rows)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'module_sha256':hashlib.sha256((runtime/'model.imasm').read_bytes()).hexdigest()}
    if vm:
        report['vm']=[]
        for bits,(q,expected,priority) in observed.items():
            args=','.join(map(str,(*words(q),3)))
            command=[str(root/'Vox/target/release/vox'),'run','vgi_check_execute',str(runtime/'model.imasm'),'--args',args]
            result=subprocess.run(command,capture_output=True,timeout=180);text=(result.stdout+result.stderr).decode()
            (runtime/f'checking_vm_shape_{bits}.log').write_text(text)
            match=re.search(r'\)\s*=\s*(-?\d+)',text)
            assert result.returncode==0 and match and int(match[1])==expected,text
            count_command=command[:-1]+[','.join(map(str,(*words(q),2)))]
            result=subprocess.run(count_command,capture_output=True,timeout=180);count_text=(result.stdout+result.stderr).decode()
            (runtime/f'checking_vm_shape_{bits}_calls.log').write_text(count_text)
            match=re.search(r'\)\s*=\s*(-?\d+)',count_text)
            assert result.returncode==0 and match and int(match[1])==bits&1,count_text
            report['vm'].append({'predicted_shape':bits,'packed_return':expected,'actual_oracle_calls':int(match[1]),
                                 'proposal':q[9]-266,'counterfactual_priority':priority,'passed':True})
    (runtime/f'checking_predictions_{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
    (runtime/f'checking_evaluation_{split}.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','runtime','tasks'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--split',choices=('validation','test'),default='validation');p.add_argument('--vm',action='store_true')
    a=p.parse_args();verify(a.root.resolve(),a.runtime.resolve(),a.tasks.resolve(),a.split,a.vm)
