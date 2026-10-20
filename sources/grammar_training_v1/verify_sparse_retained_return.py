#!/usr/bin/env python3
"""Compare sparse returns to every-record returns and recover an interrupted capsule."""
import argparse,hashlib,json,subprocess
from pathlib import Path
from retained_format import decode_capsules


def verify(runtime,control,profile):
    build=json.loads((runtime/'build.json').read_text());other=json.loads((control/'build.json').read_text())
    assert build['parent']==other['parent'] and build['corpus_oracle']==other['corpus_oracle']
    assert build['snapshot_records']>1 and other['snapshot_records']==1
    with (control/'control.stdout').open('wb') as out,(control/'control.stderr').open('wb') as err:
        process=subprocess.run([str(control/'model')],stdout=out,stderr=err,timeout=180)
    assert process.returncode==0
    sparse=profile.with_suffix('.stdout').read_bytes();dense=(control/'control.stdout').read_bytes()
    a,tail=decode_capsules(sparse);assert not tail
    b,tail=decode_capsules(dense);assert not tail
    expected={state['completed_records']:(weights,state) for weights,state in b}
    for index,(weights,state) in enumerate(a):
        count=state['completed_records']
        # The last dense return also includes settlement. Intermediate returns
        # retain the pre-settlement carrier even if their record is the last one.
        if index==len(a)-1:assert (weights,state)==b[-1]
        else:
            candidates=[(w,s) for w,s in b if s['completed_records']==count and s['settlement_code']==0]
            assert (weights,state) in candidates
    assert a[-1][0]==b[-1][0] and a[-1][1]==b[-1][1]
    # End during the final complete-state capsule, preserving the prior snapshot.
    last_weights,last_state=a[-1]
    final_size=len(last_weights)+128+32*last_state['width']+2*len(last_state['source_bank'])
    cut=len(sparse)-final_size+min(64,final_size-1)
    recovered,trailing=decode_capsules(sparse[:cut],allow_partial=True)
    assert len(recovered)==len(a)-1 and trailing
    assert recovered[-1]==a[-2]
    report={'status':'passed','sparse_capsules':len(a),'dense_capsules':len(b),
            'records':last_state['completed_records'],'checkpoint_interval':build['snapshot_records'],
            'all_sparse_returns_match_dense_controls':True,'final_weights_and_source_state_identical':True,
            'interrupted_final_return_recovers_records':recovered[-1][1]['completed_records'],
            'sparse_stdout_sha256':hashlib.sha256(sparse).hexdigest(),
            'dense_stdout_sha256':hashlib.sha256(dense).hexdigest()}
    (runtime/'sparse_return_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('runtime','control','profile'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();verify(a.runtime.resolve(),a.control.resolve(),a.profile.resolve())
