#!/usr/bin/env python3
"""Extend the existing full learner with shared neural graph execution and its source oracle."""
import argparse,hashlib,json,re,subprocess
from pathlib import Path
from export_full_learning import export,compile_model


def build(root,parent,out,library,vocabulary,mode=0,steps=0,passes=0,optimizer='sgd'):
    export(root,parent,out,retained=True,retained_settle=False,context_length=16,optimization='O2',graph_library=library,graph_vocabulary=vocabulary,optimizer=optimizer)
    table=json.loads(vocabulary.read_text())['tokens'];assert [r['id'] for r in table]==list(range(len(table)))
    source=out/'full_model.cpp';body=source.read_text();body=body[:body.rindex('extern "C" __attribute__((force_align_arg_pointer)) void _start()')]
    body+='\nextern "C" __attribute__((force_align_arg_pointer)) void _start(){vgi_retained_reset();'
    body+=f'long result=vgi_graph_stream({mode},{steps},{passes},0);long status=result<0?2:0;'
    body+='if(vgi_write_current()<0||retained_write_state()<0||graph_write_state()<0)status=3;'
    body+='__asm__ volatile("syscall"::"a"(60L),"D"(status):"rcx","r11","memory");__builtin_unreachable();}\n'
    source.write_text(body)
    report=json.loads((out/'build.json').read_text());base_commands=report['commands'];commands=[]
    commands=compile_model(root,source,out,'O2',library)
    report.update(graph_learning={'mode':mode,'correction_steps':steps,'passes':passes,'library':str(library),
        'library_sha256':hashlib.sha256(library.read_bytes()).hexdigest(),
        'vocabulary':str(vocabulary),'vocabulary_sha256':hashlib.sha256(vocabulary.read_bytes()).hexdigest(),
        'query_format':'shared retained eight-token local port operator','entry':'vgi_graph_stream'},
        base_export_commands=base_commands,commands=commands,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        module_sha256=hashlib.sha256((out/'model.imasm').read_bytes()).hexdigest())
    (out/'build.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'built':str(out),'graph_mode':mode}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','checkpoint','out','library','vocabulary'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--mode',type=int,choices=(0,1,2,3),default=0);p.add_argument('--steps',type=int,default=0);p.add_argument('--passes',type=int,default=0)
    p.add_argument('--optimizer',choices=('sgd','adam'),default='sgd')
    a=p.parse_args();build(a.root.resolve(),a.checkpoint.resolve(),a.out.resolve(),a.library.resolve(),a.vocabulary.resolve(),a.mode,a.steps,a.passes,a.optimizer)
