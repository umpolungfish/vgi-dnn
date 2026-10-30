#!/usr/bin/env python3
"""Run one structured distinction query through the executing Vox model."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from distinction_format import tokens,words


def run(root,runtime,query):
    first,second=words(tokens(query))
    command=[str(root/'Vox/target/release/vox'),'run','vgi_policy_act',str(runtime/'model.imasm'),
             '--args',f'{first},{second},5']
    process=subprocess.run(command,capture_output=True,text=True,timeout=60,check=True)
    observation=process.stdout+process.stderr
    match=re.search(r'\)\s*=\s*(-?\d+)',observation)
    if not match or int(match[1])<0:raise RuntimeError('query was not admitted: '+observation)
    packed=int(match[1]);action=(packed&7)-1;carrier=(packed>>3)&15;evidence=(packed>>7)&255
    names=('direct','split_two','split_three','release')
    result={'query':query,'action':names[action] if action>=0 else 'open',
            'returned_carrier':{'bits':carrier,'atoms':[a for i,a in enumerate(('T','F','t','f')) if carrier&(1<<i)]},
            'candidate_evidence':{name:('N','T','F','B')[(evidence>>(2*i))&3] for i,name in enumerate(names)},
            'revised':bool(packed&(1<<15)),'released':bool(packed&(1<<16)),
            'command':command,'observation':observation,
            'module_sha256':hashlib.sha256((runtime/'model.imasm').read_bytes()).hexdigest()}
    print(json.dumps(result,ensure_ascii=False,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','runtime','query'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--out',type=Path)
    a=p.parse_args();result=run(a.root.resolve(),a.runtime.resolve(),json.loads(a.query.read_text()))
    if a.out:
        with a.out.open('x',encoding='utf8') as handle:handle.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
