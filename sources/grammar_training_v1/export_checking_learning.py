#!/usr/bin/env python3
"""Lift the shared learner with complete checking return and continuation."""
import argparse,hashlib,json
from pathlib import Path
from export_full_learning import export,compile_model

def build(root,checkpoint,out,optimizer='adam'):
    export(root,checkpoint,out,retained=True,context_length=16,optimization='O2',optimizer=optimizer)
    source=out/'full_model.cpp';body=source.read_text();body=body[:body.rindex('extern "C" __attribute__((force_align_arg_pointer)) void _start()')]
    body+='''
extern "C" __attribute__((force_align_arg_pointer)) void _start(){
    vgi_retained_reset();long result=vgi_check_stream();long status=result<0?2:0;
    if(!status&&vgi_check_return()<0)status=3;
    __asm__ volatile("syscall"::"a"(60L),"D"(status):"rcx","r11","memory");__builtin_unreachable();}
'''
    source.write_text(body);report=json.loads((out/'build.json').read_text())
    report['base_export_commands']=report['commands'];report['commands']=compile_model(root,source,out,'O2')
    report.update(checking_entry='vgi_check_stream',checking_protocol='CGRUN1 / CGSTATE2 with retained VGIRM2',
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  module_sha256=hashlib.sha256((out/'model.imasm').read_bytes()).hexdigest())
    (out/'build.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'built':str(out),'checking_return':True}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','checkpoint','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--optimizer',choices=('sgd','adam'),default='adam')
    a=p.parse_args();build(a.root.resolve(),a.checkpoint.resolve(),a.out.resolve(),a.optimizer)
