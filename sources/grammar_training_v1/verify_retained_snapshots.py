#!/usr/bin/env python3
"""Exercise full and interrupted record-capsule returns from the baked learner."""
import argparse,hashlib,json,subprocess
from pathlib import Path
from retained_format import decode_capsules,stream,pack,MAGIC
import struct

def verify(runtime,fixture,baseline):
    rows=[json.loads(l) for l in (fixture/'records.jsonl').open()]
    checks=[]
    for label,payload,status,count in [('complete',stream(rows),0,len(rows)),
            ('partial_record',MAGIC+struct.pack('<I',2)+pack(rows[0])+pack(rows[1])[:9],2,1)]:
        result=subprocess.run([str(runtime/'model')],input=payload,capture_output=True,timeout=60)
        assert result.returncode==status,result.stderr
        (runtime/(label+'.stdout')).write_bytes(result.stdout)
        (runtime/(label+'.stderr')).write_bytes(result.stderr)
        capsules,tail=decode_capsules(result.stdout)
        assert not tail and len(capsules)==count+1
        for index,(weights,state) in enumerate(capsules[:-1]):
            assert state['completed_records']==index+1 and state['record_complete']==1
            assert state['source_bank']==rows[index]['prefix']+rows[index]['target']
            assert state['settlement_code']==0
        assert capsules[-1][1]['completed_records']==count
        assert capsules[-1][1]['source_bank']==rows[count-1]['prefix']+rows[count-1]['target']
        if label=='complete':
            assert capsules[-1][0]==(baseline/'model.safetensors').read_bytes()
            expected=json.loads((baseline/'retained_state.json').read_text())
            assert capsules[-1][1]==expected
        else:assert capsules[0][0]==capsules[-1][0]
        # Every byte offset during the last capsule recovers the committed prefix.
        last_start=len(result.stdout)-len(capsules[-1][0])-128-32*capsules[-1][1]['width']-2*len(capsules[-1][1]['source_bank'])
        for cut in (1,7,8,64,len(result.stdout)-last_start-1):
            prefix,trailing=decode_capsules(result.stdout[:last_start+cut],allow_partial=True)
            assert len(prefix)==count and len(trailing)==cut
        checks.append({'case':label,'records':count,'capsules':len(capsules),'exit_status':status,
                       'stdout_sha256':hashlib.sha256(result.stdout).hexdigest()})
    receipt={'status':'passed','execution':'standalone CPU execution of Vox-lifted construction',
             'checks':checks,'complete_weights_and_settlement_match_prior_verified_return':True,
             'partial_record_retains_committed_weights':True,'partial_capsule_recovers_complete_prefix':True}
    (runtime/'snapshot_verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('runtime','fixture','baseline'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();verify(a.runtime.resolve(),a.fixture.resolve(),a.baseline.resolve())
