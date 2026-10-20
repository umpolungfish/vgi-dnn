#!/usr/bin/env python3
"""Compile the canonical graph gate into a freestanding, Vox-lifted binding."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path


def build(root,out):
    out.mkdir(parents=True,exist_ok=False);temporary=out/'buildtmp';temporary.mkdir()
    manifest=root/'VGI/sources/graph_flow_oracle/Cargo.toml';target='x86_64-unknown-linux-gnu'
    commands=[]
    def call(command,env=None):
        commands.append(command);subprocess.run(command,check=True,env=env)
    for mode,extra in [('bare',['--lib']),('host',['--features','host','--bin','graph_flow_oracle'])]:
        environment=dict(os.environ,CARGO_TARGET_DIR=str(out/mode),TMPDIR=str(temporary))
        call(['cargo','build','--offline','--release','--target',target,'--manifest-path',str(manifest),*extra],environment)
    library=out/'bare'/target/'release/libig_graph_flow_oracle.a'
    source=out/'probe.c';source.write_text('''extern unsigned int ig_flow_gate(unsigned int,unsigned int,unsigned int,unsigned int,unsigned int);
long vgi_graph_gate(long op,long input,long seed,long slot,long fan){return ig_flow_gate(op,input,seed,slot,fan);}
void _start(void){__asm__ volatile("syscall"::"a"(60L),"D"(0L):"rcx","r11","memory");__builtin_unreachable();}
''')
    environment=dict(os.environ,TMPDIR=str(temporary))
    call(['cc','-O2','-fno-stack-protector','-ffunction-sections','-fdata-sections','-nostdlib','-static','-no-pie',
          '-Wl,--gc-sections','-Wl,-e,_start','-Wl,-u,vgi_graph_gate',str(source),str(library),'-o',str(out/'probe')],environment)
    call(['cc','-O2','-fPIC','-shared','-Wl,-u,ig_flow_gate',str(source),str(library),'-o',str(out/'probe.so')],environment)
    call([str(root/'Vox/target/release/vox'),'imasm',str(out/'probe'),'--output',str(out/'probe.imasm')])
    host=out/'host'/target/'release/graph_flow_oracle'
    sources=[root/'MoDoT/imasm_core/src'/n for n in ('flow.rs','check.rs','classic.rs','imasm16_3.rs')]
    sources.extend((manifest,manifest.parent/'src/lib.rs',manifest.parent/'src/bin/graph_flow_oracle.rs'))
    report={'commands':commands,'host':str(host),'library':str(library),'probe':str(out/'probe'),
            'source_bindings':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            'probe_module_sha256':hashlib.sha256((out/'probe.imasm').read_bytes()).hexdigest(),
            'boundary':'existing imasm_core::flow::gate_out linked through a retained executable symbol'}
    (out/'build.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();build(a.root.resolve(),a.out.resolve())
