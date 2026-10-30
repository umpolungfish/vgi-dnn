#!/usr/bin/env python3
"""Retrieve a source coordinate through Vox using its complete corpus identity."""
import argparse,hashlib,json,re,subprocess
from pathlib import Path


def invoke(root,runtime,symbol,args):
    # Vox's CLI takes signed i64 arguments; preserve every bit of unsigned keys.
    signed=[value if value<(1<<63) else value-(1<<64) for value in args]
    command=[str(root/'Vox/target/release/vox'),'run',symbol,str(runtime/'model.imasm'),
             '--args',','.join(map(str,signed))]
    result=subprocess.run(command,capture_output=True,text=True,timeout=90)
    observation=result.stdout+result.stderr;match=re.search(r'\)\s*=\s*(-?\d+)',observation)
    if result.returncode or not match:raise RuntimeError(observation)
    return int(match[1]),{'command':command,'observation':observation}


def retrieve(root,runtime,identity,position):
    digest=hashlib.sha256(identity.encode()).digest()
    keys=[int.from_bytes(digest[i:i+8],'little') for i in range(0,32,8)]
    index,lookup=invoke(root,runtime,'vgi_corpus_find',keys)
    if index<0:
        return {'example_id':identity,'lookup_sha256':digest.hex(),'status':'unselected','lookup_return':index,'lookup':lookup}
    value,readback=invoke(root,runtime,'vgi_corpus_retrieve',[index,position,0])
    report={'example_id':identity,'lookup_sha256':digest.hex(),'record_index':index,
            'position':position,'value':value,'status':'returned' if value>=0 else 'unselected',
            'lookup':lookup,'readback':readback,
            'module_sha256':hashlib.sha256((runtime/'model.imasm').read_bytes()).hexdigest()}
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--id',required=True,help='complete source example ID; every SHA256 bit participates in lookup')
    p.add_argument('--position',type=int,required=True,help='coordinate index, or -1 for complete record length')
    p.add_argument('--out',type=Path)
    a=p.parse_args();report=retrieve(a.root.resolve(),a.runtime.resolve(),a.id,a.position)
    if a.out:a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
