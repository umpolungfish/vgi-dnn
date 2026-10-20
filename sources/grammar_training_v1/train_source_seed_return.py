#!/usr/bin/env python3
"""Exercise the shared learner's source boundary over every four-atom seed."""
import argparse,ctypes,hashlib,json,os,random,shutil
from pathlib import Path


def train(runtime,out,epochs,steps):
    out.mkdir(parents=True,exist_ok=False)
    lib=ctypes.CDLL(str(runtime/'model.so'));lib.vgi_retained_reset()
    lib.vgi_retained_token.argtypes=[ctypes.c_long]*2;lib.vgi_retained_token.restype=ctypes.c_long
    lib.vgi_retained_update.argtypes=[ctypes.c_long]*6;lib.vgi_retained_update.restype=ctypes.c_long
    lib.vgi_retained_first.argtypes=[ctypes.c_long]*2;lib.vgi_retained_first.restype=ctypes.c_double
    lib.ig_flow_gate.argtypes=[ctypes.c_uint]*5;lib.ig_flow_gate.restype=ctypes.c_uint
    build=json.loads((runtime/'build.json').read_text());parent=Path(build['parent'])
    def query(seed,updates):
        tokens=[256,71,0,266,266+seed,1,0,257]
        for i,t in enumerate(tokens):assert lib.vgi_retained_token(i,t)>=0
        expected=lib.ig_flow_gate(0,0,seed,0,1);assert expected==seed
        assert lib.vgi_retained_update(8,0,2,expected if updates else 0,updates,1)>=0
        predicted=sum(int(lib.vgi_retained_first(k%2,259+k//2)>0)<<k for k in range(4))
        return predicted
    def observe(epoch):
        predicted=[query(seed,0) for seed in range(16)]
        row={'epoch':epoch,'seeds':list(range(16)),'predicted':predicted,
             'exact':sum(seed==value for seed,value in enumerate(predicted))}
        print(json.dumps(row),flush=True);return row
    history=[observe(0)];records=[];rng=random.Random(73)
    for epoch in range(1,epochs+1):
        seeds=list(range(16));rng.shuffle(seeds)
        for seed in seeds:
            before=query(seed,0);applied=steps if before!=seed else 0
            query(seed,applied);after=query(seed,0)
            records.append({'epoch':epoch,'seed':seed,'before':before,'after':after,'steps':applied,
                            'target_source':'linked imasm_core::flow::gate_out VINIT seed return'})
        history.append(observe(epoch))
    checkpoint=out/'returned_checkpoint';checkpoint.mkdir()
    with (checkpoint/'model.safetensors').open('wb') as handle:
        saved=os.dup(1)
        try:os.dup2(handle.fileno(),1);result=lib.vgi_write_current()
        finally:os.dup2(saved,1);os.close(saved)
    assert result>=0
    for name in ('config.json','tokenizer.json','execution_input.json'):
        if (parent/name).exists():shutil.copy2(parent/name,checkpoint/name)
    report={'parent':str(parent),'parent_checkpoint_sha256':hashlib.sha256((parent/'model.safetensors').read_bytes()).hexdigest(),
            'runtime':str(runtime),'runtime_sha256':hashlib.sha256((runtime/'model.so').read_bytes()).hexdigest(),
            'epochs':epochs,'max_steps_per_mismatch':steps,'learning_rate':build['learning_rate'],
            'learning_steps':sum(r['steps'] for r in records),'history':history,
            'checkpoint_sha256':hashlib.sha256((checkpoint/'model.safetensors').read_bytes()).hexdigest(),
            'execution':'shared exported CPU operator; source-only seed boundary curriculum',
            'corpus_integration':'candidate checkpoint; main campaign retains its own uninterrupted lineage'}
    (out/'updates.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('runtime','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--epochs',type=int,default=16);p.add_argument('--steps',type=int,default=32)
    a=p.parse_args()
    if a.epochs<1 or not 1<=a.steps<=100:p.error('positive epochs and 1..100 steps required')
    train(a.runtime.resolve(),a.out.resolve(),a.epochs,a.steps)
