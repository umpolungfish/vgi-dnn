#!/usr/bin/env python3
"""Compare an executing source-port correction's parameter return through Vox."""
import argparse,ctypes,json,os,re,subprocess
from pathlib import Path


def verify(root,runtime,budget):
    lib=ctypes.CDLL(str(runtime/'model.so'));lib.vgi_corpus_learn.argtypes=[ctypes.c_long]*3;lib.vgi_corpus_learn.restype=ctypes.c_long
    lib.vgi_retained_reset();expected=lib.vgi_corpus_learn(0,1,1)
    command=[str(root/'Vox/target/release/vox'),'run','vgi_corpus_learn',str(runtime/'model.imasm'),'--args','0,1,1']
    env=dict(os.environ,VOX_STEPS=str(budget),VOX_PROGRESS='1')
    with (runtime/f'local_oracle_vm_{budget}.stderr').open('wb') as log:
        process=subprocess.run(command,capture_output=False,stdout=subprocess.PIPE,stderr=log,env=env,timeout=600)
    observation=process.stdout.decode()+(runtime/f'local_oracle_vm_{budget}.stderr').read_text()
    match=re.search(r'\)\s*=\s*(-?\d+)',observation)
    report={'command':command,'instruction_budget':budget,'exit_status':process.returncode,'expected_updated_parameter_digest':expected,
            'status':'passed' if process.returncode==0 and match and int(match[1])==expected else 'returned_without_matching_digest',
            'observation':observation}
    (runtime/f'local_oracle_vm_{budget}.json').write_text(json.dumps(report,indent=2)+'\n')
    assert report['status']=='passed',observation
    print(json.dumps({'status':'passed','instruction_budget':budget,'updated_parameter_digest':expected}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--instruction-budget',type=int,default=1000000000)
    a=p.parse_args();verify(a.root.resolve(),a.runtime.resolve(),a.instruction_budget)
