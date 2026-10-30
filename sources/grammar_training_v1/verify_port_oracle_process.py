#!/usr/bin/env python3
"""Run one complete port-oracle update through Vox's budgeted process entry."""
import argparse,hashlib,json,os,subprocess,struct
from pathlib import Path
from export_full_learning import compile_model
from retained_format import decode_return


def verify(root,runtime,out,budget,record_index=0,joint=False):
    out.mkdir(parents=True,exist_ok=True)
    parent=json.loads((runtime/'build.json').read_text())
    body=(runtime/'full_model.cpp').read_text()
    if joint and 'extern "C" long vgi_joint_corpus_learn(' not in body:
        raise ValueError('runtime has no joint source/graph update entry')
    body=body[:body.rindex('extern "C" __attribute__((force_align_arg_pointer)) void _start()')]
    body+='''
extern "C" __attribute__((force_align_arg_pointer)) void _start(){
    vgi_retained_reset();long result=vgi_corpus_learn(RECORD_INDEX,1,1);long status=result<0?2:0;
    if(vgi_write_current()<0||retained_write_state()<0)status=3;
    __asm__ volatile("syscall"::"a"(60L),"D"(status):"rcx","r11","memory");__builtin_unreachable();}
'''.replace('RECORD_INDEX',str(record_index))
    if joint:
        body=body.replace(f'vgi_corpus_learn({record_index},1,1)',f'vgi_joint_corpus_learn({record_index})')
    source=out/'full_model.cpp';source.write_text(body)
    library=Path(parent['graph_library'])
    commands=compile_model(root,source,out,'O2',library)
    native=subprocess.run([str(out/'model')],capture_output=True,timeout=90)
    assert native.returncode==0,native.stderr
    expected=decode_return(native.stdout)
    assert expected[1]['committed_updates']==1,'selected row did not execute a learning update'
    (out/'native.stdout').write_bytes(native.stdout);(out/'native.stderr').write_bytes(native.stderr)
    command=[str(root/'Vox/target/release/vox'),'run',str(out/'model.imasm')]
    with (out/'vm.stdout').open('wb') as stdout,(out/'vm.stderr').open('wb') as stderr:
        vm=subprocess.run(command,stdout=stdout,stderr=stderr,env=dict(os.environ,VOX_STEPS=str(budget),VOX_PROGRESS='1'),timeout=600)
    try:actual=decode_return((out/'vm.stdout').read_bytes())
    except (ValueError,struct.error) as error:
        report={'status':'incomplete_return','command':command,'instruction_budget':budget,'record_index':record_index,
                'exit_status':vm.returncode,'return_error':str(error),'native_update_completed':True,
                'native_committed_updates':expected[1]['committed_updates'],
                'stderr_tail':(out/'vm.stderr').read_text()[-5000:]}
        (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');raise
    assert vm.returncode==0,(out/'vm.stderr').read_text()
    assert actual==expected,'VM model or retained source differs from native correction'
    report={'status':'passed','command':command,'instruction_budget':budget,'commands':commands,'joint_source_graph_update':joint,
            'updated_checkpoint_sha256':hashlib.sha256(actual[0]).hexdigest(),
            'updated_weights_byte_exact':True,'retained_source_and_memory_exact':True,
            'committed_updates':actual[1]['committed_updates'],'target_bits':actual[1]['bits'],'record_index':record_index,
            'module_sha256':hashlib.sha256((out/'model.imasm').read_bytes()).hexdigest()}
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','runtime','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--instruction-budget',type=int,default=1000000000)
    p.add_argument('--record-index',type=int,default=0)
    p.add_argument('--joint',action='store_true',help='execute the runtime joint source/graph learner before returning the complete checkpoint')
    a=p.parse_args();verify(a.root.resolve(),a.runtime.resolve(),a.out.resolve(),a.instruction_budget,a.record_index,a.joint)
