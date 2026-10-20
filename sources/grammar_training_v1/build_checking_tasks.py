#!/usr/bin/env python3
"""Source-checked examples for choosing whether to check a candidate return."""
import argparse,ctypes,hashlib,json
from collections import Counter
from pathlib import Path
from distinction_format import tokens


def query(row,action,proposal):
    original=tokens(row)
    return [256,67,*original[1:7],290+action,266+proposal,0,257]


def words(q):
    return tuple(sum(q[i+j]<<(9*j) for j in range(4)) for i in (0,4,8))


def build(tasks,runtime,out):
    out.mkdir(parents=True,exist_ok=False)
    lib=ctypes.CDLL(str(runtime/'model.so'));lib.vgi_check_reference.argtypes=[ctypes.c_ulong]*3+[ctypes.c_long];lib.vgi_check_reference.restype=ctypes.c_long
    counts=Counter();shapes=Counter();sources={}
    for split in ('train','validation','test'):
        source=tasks/f'{split}.jsonl';sources[split]=hashlib.sha256(source.read_bytes()).hexdigest()
        ordinal=0
        with (out/f'{split}.jsonl').open('w') as output:
            for line in source.open():
                original=json.loads(line);assert original['split']==split
                for action,candidate in enumerate(original['candidates']):
                    value=candidate['return_bits']
                    for proposal in (value,value^1):
                        q=query(original,action,proposal);a,b,c=words(q)
                        carrier=lib.vgi_check_reference(a,b,c,1);bits=lib.vgi_check_reference(a,b,c,0)
                        expected=int(proposal!=value or original['refute'][action])|\
                                 (int(proposal==value and original['support'][action])<<1)
                        assert carrier==value and bits==expected
                        identifier=hashlib.sha256(json.dumps([original['id'],action,proposal]).encode()).hexdigest()
                        row={'prefix':q,'target':[],'kind':5,'bits':0,'steps':1,'checkpoint_chunks':1,
                             'example_id':identifier,'source_id':original['id'],'family':original['family'],
                             'task':'checking_policy','task_ordinal':ordinal,'training_split':split,
                             'reference_bits':bits,'checking_reference':{'action':action,'proposal':proposal,'carrier':carrier,
                                 'candidate_support':original['support'][action],'candidate_refute':original['refute'][action]},
                             'source':{'task_file':str(source),'task_sha256':sources[split],
                                       'function':'checking_reference over the existing distinction candidate operations'},
                             'criterion':'support checking on carrier mismatch or candidate refutation; refute checking on exact supported return'}
                        output.write(json.dumps(row,ensure_ascii=False)+'\n');ordinal+=1;shapes[f'{split}:{bits}']+=1
        counts[split]=ordinal
    report={'counts':dict(counts),'evidence_shapes':dict(shapes),'source_sha256':sources,
            'runtime_module_sha256':hashlib.sha256((runtime/'model.imasm').read_bytes()).hexdigest(),
            'split_grouping':'inherits seed/work families from the source distinction tasks',
            'input':'twelve query tokens including proposed action and carrier; no supplied target bits',
            'reference_verification':'every source carrier and checking mask equals independent recorded candidate criteria',
            'inference':'reference operation is called only after a learned affirmative checking decision'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('tasks','runtime','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();build(a.tasks.resolve(),a.runtime.resolve(),a.out.resolve())
