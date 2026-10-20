#!/usr/bin/env python3
"""Measure label-free local carrier proposals from an exported shared learner."""
import argparse,ctypes,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path


def evaluate(runtime,records):
    lib=ctypes.CDLL(str(runtime/'model.so'));lib.vgi_retained_reset()
    lib.vgi_retained_token.argtypes=[ctypes.c_long]*2;lib.vgi_retained_token.restype=ctypes.c_long
    lib.vgi_retained_update.argtypes=[ctypes.c_long]*6;lib.vgi_retained_update.restype=ctypes.c_long
    lib.vgi_retained_first.argtypes=[ctypes.c_long]*2;lib.vgi_retained_first.restype=ctypes.c_double
    lib.ig_flow_gate.argtypes=[ctypes.c_uint]*5;lib.ig_flow_gate.restype=ctypes.c_uint
    predictions=[];counts=defaultdict(Counter);confusion=Counter()
    for line in records.open():
        row=json.loads(line);query=row['prefix']
        for i,token in enumerate(query):assert lib.vgi_retained_token(i,token)>=0
        # The proposal receives no expected carrier bits and makes no update.
        assert lib.vgi_retained_update(8,0,2,0,0,1)>=0
        predicted=sum(int(lib.vgi_retained_first(k%2,259+k//2)>0)<<k for k in range(4))
        expected=lib.ig_flow_gate(query[2],query[3]-266,query[4]-266,query[6],query[5])
        assert expected==row['bits']
        group=counts[row['operation']['name']];group['cases']+=1;group['exact']+=predicted==expected
        group['correct_bits']+=4-(predicted^expected).bit_count();confusion[expected,predicted]+=1
        predictions.append({'id':row['example_id'],'expected':expected,'predicted':predicted})
    cases=sum(c['cases'] for c in counts.values());exact=sum(c['exact'] for c in counts.values())
    report={'cases':cases,'exact':exact,'accuracy':exact/cases,'per_operation':dict(counts),
            'confusion':[{'expected':a,'predicted':b,'cases':n} for (a,b),n in sorted(confusion.items())],
            'proposal_target_bits':0,'learning_steps':0,'evaluation_domain':'complete local gate table; graph structures evaluated separately',
            'parent':json.loads((runtime/'build.json').read_text())['parent'],
            'module_sha256':hashlib.sha256((runtime/'model.imasm').read_bytes()).hexdigest(),
            'records_sha256':hashlib.sha256(records.read_bytes()).hexdigest()}
    (runtime/'local_port_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
    (runtime/'local_port_evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('cases','exact','accuracy','per_operation')}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('runtime','records'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();evaluate(a.runtime.resolve(),a.records.resolve())
