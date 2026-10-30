#!/usr/bin/env python3
"""Compile reverse-mode DT learning into a self-contained Vox IMASM module."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import torch
from safetensors.torch import load_file

from export_imasm_model import block_fields
from optimizer_state import split_state


def export(root, checkpoint, output, bits=3, steps=1, unroll=0, objective='carrier', learning_rate=.001,
           stream=False, context_length=None, gradient_clip=1.,retained=False,retained_settle=False,optimization='O1',snapshot_records=0,corpus_records=None,corpus_entry=False,graph_library=None,graph_vocabulary=None,optimizer='sgd'):
    if snapshot_records<0 or (snapshot_records and not retained):raise ValueError("snapshots require retained mode and a nonnegative interval")
    if (corpus_records or corpus_entry) and not retained:raise ValueError("corpus oracle requires retained mode")
    if corpus_entry and not corpus_records:raise ValueError("corpus entry requires source records")
    if bool(graph_library)!=bool(graph_vocabulary) or (graph_library and not retained):raise ValueError('graph library and vocabulary require retained mode together')
    graph_table=json.loads(graph_vocabulary.read_text())['tokens'] if graph_vocabulary else None
    if graph_table and [r['id'] for r in graph_table]!=list(range(len(graph_table))):raise ValueError('graph vocabulary IDs must bind the complete source enum')
    output.mkdir(parents=True,exist_ok=False)
    state,optimizer_state=split_state(load_file(str(checkpoint/'model.safetensors')))
    if optimizer not in ('sgd','adam'):raise ValueError('unknown optimizer')
    if optimizer_state and optimizer!='adam':raise ValueError('Adam checkpoint requires Adam continuation')
    cfg=json.loads((checkpoint/'config.json').read_text())
    probe=json.loads((checkpoint/'execution_input.json').read_text())
    context_length=context_length or len(probe['tokens'])
    if not len(probe['tokens'])<=context_length<=cfg['max_len']:
        raise ValueError('context length must contain the probe and fit model positions')
    probe_tokens=probe['tokens']+[0]*(context_length-len(probe['tokens']))
    template=Path(__file__).with_name('dialetheic_runtime.c').read_text()
    math=template[template.index('static double ex'):template.index('typedef struct {double')]
    math=re.sub(r'\b(ex|ln|sq|sig|lg)\b',r'raw_\1',math)
    core=template[template.index('typedef struct {double'):template.index('/* Individual fixed-point')]
    core=core[:core.index('static void settle(')]+core[core.index('static void forward('):]
    core=core.replace('if(reentry)settle(&x);','for(int r=0;r<reentry;r++)block(&x,LAYERS-1);')
    core=core.replace('double','Var')
    tape_capacity=max(1000000,context_length*128000*(cfg['n_layers']+2))
    lines=[f'#define VGI_TAPE_CAPACITY {tape_capacity}',math,Path(__file__).with_name('reverse_tape.hpp').read_text()]
    defs={'V':cfg['vocab_size'],'D':cfg['d_model'],'H':cfg['n_heads'],'FF':cfg['d_ff'],
          'LAYERS':cfg['n_layers'],'MAXLEN':cfg['max_len'],'TOKENS':context_length,
          'BLEND':int(cfg['residual']=='blend'),'CAUSAL':int(cfg['causal'])}
    defs.update({'LOSS_BITS':8 if objective=='policy' else 4,'LOSS_BASE':261 if objective=='policy' else 259})
    if not 0<learning_rate<1:raise ValueError('learning rate must be between zero and one')
    defs['RETAINED_SNAPSHOT_RECORDS']=snapshot_records
    defs['LEARNING_RATE']=format(learning_rate,'.17g')
    if gradient_clip<0:raise ValueError('gradient clip must be nonnegative')
    defs['STREAM_CLIP']=format(gradient_clip,'.17g')
    if graph_table:defs['GRAPH_TOKEN_COUNT']=len(graph_table)
    lines.extend(f'#define {key} {value}' for key,value in defs.items())
    if graph_table:
        for name,key in [('graph_arity_in','input_ports'),('graph_arity_out','output_ports')]:
            lines.append('static const long '+name+'[]={'+','.join(str(row[key]) for row in graph_table)+'};')
    lines.append('static int tokens[]={' + ','.join(map(str,probe_tokens))+'};')
    mapping={'E_t':'embedding_t','E_f':'embedding_f','pe':'position','final_norm.gamma':'final_norm',
             'head.A':'headA','head.B':'headB','head.b':'headb'}
    fields=block_fields()
    for l in range(cfg['n_layers']):
        mapping.update({f'blocks.{l}.{key}':f'b{l}_{name}' for name,key in fields})
    metadata=[];trainable=[];buffers=[];header={};offset=0
    for i,(key,tensor) in enumerate(sorted(state.items())):
        raw=f'raw_{i}';values=tensor.flatten().tolist()
        boolean=tensor.dtype==torch.bool
        dtype='BOOL' if boolean else 'F64';size=len(values)*(1 if boolean else 8)
        if boolean:
            lines.append(f'static unsigned char {raw}[]={{'+','.join(str(int(v)) for v in values)+'};')
        else:
            name=mapping[key]
            literal=','.join(format(float(v),'.17g') for v in values)
            lines.append(f'static const double initial_{i}[]={{'+literal+'};')
            lines.append(f'static double {raw}[{len(values)}];static Var {name}[{len(values)}];')
            entry={'key':key,'raw':raw,'initial':f'initial_{i}','var':name,'size':len(values)}
            (buffers if key=='pe' else trainable).append(entry)
        header[key]={'dtype':dtype,'shape':list(tensor.shape),'data_offsets':[offset,offset+size]}
        metadata.append((raw,size));offset+=size
    param_count=sum(e['size'] for e in trainable)
    lines.append(f'#define PARAMETERS {param_count}')
    lines.append('static double initial_gradient[PARAMETERS],optimizer_gradient[PARAMETERS];')
    lines.append('struct Parameter {double*raw;const double*initial;Var*var;int size;};')
    for label,entries in [('parameters',trainable),('buffers',buffers)]:
        lines.append('static const Parameter '+label+'[]={'+','.join(
            '{'+f'{e["raw"]},{e["initial"]},{e["var"]},{e["size"]}'+'}' for e in entries)+'};')
    lines.append('typedef struct {'+''.join('const Var *'+n+';' for n,k in fields)+'} BlockWeights;')
    lines.append('static const BlockWeights blocks[]={'+','.join(
        '{'+','.join(f'b{l}_{name}' for name,key in fields)+'}' for l in range(cfg['n_layers']))+'};')
    lines.append(core)
    lines.append(f'#define OPTIMIZER_ADAM {int(optimizer=="adam")}')
    if optimizer_state:
        if set(optimizer_state)!={'m','v','powers','clock'}:raise ValueError('incomplete optimizer state')
        if list(optimizer_state['m'].shape)!=[param_count] or list(optimizer_state['v'].shape)!=[param_count] or list(optimizer_state['powers'].shape)!=[2] or list(optimizer_state['clock'].shape)!=[1]:raise ValueError('optimizer state shape mismatch')
        if any(not bool(v.isfinite().all()) for v in optimizer_state.values()) or bool((optimizer_state['v']<0).any()):raise ValueError('invalid optimizer state')
        if optimizer_state['clock'].dtype!=torch.int64 or int(optimizer_state['clock'][0])<0:raise ValueError('invalid optimizer clock')
        if bool(((optimizer_state['powers']<0)|(optimizer_state['powers']>1)).any()):raise ValueError('invalid optimizer powers')
    clock=int(optimizer_state['clock'][0]) if optimizer_state else 0
    lines.append(f'static const long optimizer_initial_clock={clock};')
    for name,values in [('m',optimizer_state['m'].tolist() if optimizer_state else [0.]*param_count),
                        ('v',optimizer_state['v'].tolist() if optimizer_state else [0.]*param_count),
                        ('powers',optimizer_state['powers'].tolist() if optimizer_state else [1.,1.])]:
        lines.append('static const double optimizer_initial_'+name+'[]={'+','.join(format(float(v),'.17g') for v in values)+'};')
    lines.append(Path(__file__).with_name('optimizer_adam.hpp').read_text())
    lines.append(r'''
static int parameters_initialized;
static void reset_parameters(){
    parameters_initialized=1;
    for(const auto &p:parameters)for(int j=0;j<p.size;j++)p.raw[j]=p.initial[j];
    for(const auto &p:buffers)for(int j=0;j<p.size;j++)p.raw[j]=p.initial[j];
    optimizer_reset();
}
static void prepare(){
    used=0;overflowed=0;
    for(const auto &p:parameters)for(int j=0;j<p.size;j++)p.var[j]=leaf(p.raw[j]);
    for(const auto &p:buffers)for(int j=0;j<p.size;j++)p.var[j]=Var(p.raw[j]);
}
static Var loss_for(long bits){
    Var loss=0;
    for(int k=0;k<LOSS_BITS;k++){
        Var z=(k%2)?out_f[TOKENS-1][LOSS_BASE+k/2]:out_t[TOKENS-1][LOSS_BASE+k/2];
        /* Stable logistic loss: softplus(z)-target*z. */
        Var soft=(z.v>0)?z+ln(1+ex(-z)):ln(1+ex(z));
        loss+=soft-((bits>>k)&1)*z;
    }return loss/LOSS_BITS;
}
static long parameter_digest(){
    unsigned long hash=2166136261;
    for(const auto &p:parameters)for(int j=0;j<p.size;j++)
        hash=((hash^(unsigned long)(long)(p.raw[j]*1000000))*16777619)&0x7fffffff;
    return (long)hash;
}
extern "C" long vgi_full_learn(long bits,long steps,long unroll,long readout){
    if(bits<0||bits>((1L<<LOSS_BITS)-1)||steps<0||steps>100||unroll<0||unroll>8)return -999999999;
    reset_parameters();double before=0,after=0;
    for(long step=0;step<=steps;step++){
        prepare();forward((int)unroll);Var loss=loss_for(bits);
        if(overflowed)return -999999998;
        if(step==0)before=loss.v;after=loss.v;
        if(step<steps||step==0){
            backward(loss);int index=0;
            for(const auto &p:parameters)for(int j=0;j<p.size;j++){
                double gradient=tape[p.var[j].id].grad;
                optimizer_gradient[index]=gradient;if(step==0)initial_gradient[index]=gradient;index++;
            }
            if(step<steps)optimizer_apply(optimizer_gradient,1);
        }
    }
    if(readout==0)return (long)(before*1000000);
    if(readout==1)return (long)(after*1000000);
    if(readout==2)return parameter_digest();
    if(readout==3)return used;
    return -999999999;
}
extern "C" double vgi_full_parameter(long index){
    for(const auto &p:parameters){if(index<p.size)return p.raw[index];index-=p.size;}return 0;
}
extern "C" double vgi_full_gradient(long index){return (index>=0&&index<PARAMETERS)?initial_gradient[index]:0;}
static long checkpoint_write(const void *buffer,unsigned long length){
    unsigned long total=0;
    while(total<length){long result;const unsigned char*position=(const unsigned char*)buffer+total;
        __asm__ volatile("syscall":"=a"(result):"a"(1L),"D"(1L),"S"(position),"d"(length-total):"rcx","r11","memory");
        if(result<=0)return -1;total+=(unsigned long)result;
    }return (long)total;
}
''')
    if optimizer=='adam':
        for key,dtype,shape,raw,size in [('m','F64',[param_count],'optimizer_m',8*param_count),
                                        ('v','F64',[param_count],'optimizer_v',8*param_count),
                                        ('powers','F64',[2],'optimizer_powers',16),('clock','I64',[1],'&optimizer_clock',8)]:
            header['__optimizer.'+key]={'dtype':dtype,'shape':shape,'data_offsets':[offset,offset+size]}
            metadata.append((raw,size));offset+=size
    header['__metadata__']={'architecture':'dialetheic_transformer','reentry':'shared_final_trained_block','optimizer':optimizer,
        'update':('retained_source_reverse_mode_' if retained else 'streaming_reverse_mode_' if stream else 'full_network_reverse_mode_')+optimizer,
        'parent_sha256':hashlib.sha256((checkpoint/'model.safetensors').read_bytes()).hexdigest()}
    encoded=json.dumps(header,separators=(',',':')).encode();encoded+=b' '*(-len(encoded)%8)
    lines.append(f'static const unsigned long header_size={len(encoded)};')
    lines.append('static const unsigned char header[]={'+','.join(map(str,encoded))+'};')
    lines.append('extern "C" long vgi_write_current(){long total=0,n;')
    for name,size in [('&header_size',8),('header',len(encoded))]+metadata:
        lines.append(f'n=checkpoint_write({name},{size});if(n<0)return -1;total+=n;')
    lines.append('return total;}')
    lines.append('extern "C" long vgi_full_export(long bits,long steps,long unroll){'+
                 'long r=vgi_full_learn(bits,steps,unroll,2);if(r<0)return r;return vgi_write_current();}')
    if retained and not graph_library:
        lines.append('static long retained_graph_local_validate(const unsigned short*){return -58;} static long retained_graph_local_record(long,long){return -58;}')
    if stream or retained:lines.append(Path(__file__).with_name('stream_learning.hpp').read_text())
    if retained:
        lines.append(Path(__file__).with_name('retained_learning.hpp').read_text())
        lines.append(Path(__file__).with_name('retained_io.hpp').read_text())
        # Share the executing gate definition with the established graph policy.
        lines.append(template[template.index('static int policy_gate('):template.index('static long policy_evidence(')])
        lines.append(Path(__file__).with_name('retained_oracle.hpp').read_text())
        lines.append(Path(__file__).with_name('checking_policy.hpp').read_text())
        lines.append(Path(__file__).with_name('checking_return.hpp').read_text())
    if graph_library:lines.append(Path(__file__).with_name('graph_learning.hpp').read_text())
    corpus_report=None
    if corpus_records:
        from corpus_oracle import generate
        corpus_source,corpus_report=generate(corpus_records,cfg['vocab_size']);lines.append(corpus_source)
    settle='if(status==0)vgi_retained_settle(1);' if retained_settle else ''
    entry=('long status=vgi_retained_train()<0?2:0;'+settle+'long r=vgi_write_current();if(r<0||retained_write_state()<0)status=3;' if retained else
           'long status=vgi_stream_train()<0?2:0;long r=vgi_write_current();if(r<0)status=3;' if stream else
           f'long r=vgi_full_export({bits},{steps},{unroll});long status=r<0?2:0;')
    lines.append('extern "C" __attribute__((force_align_arg_pointer)) void _start(){'+
        entry+'__asm__ volatile("syscall"::"a"(60L),"D"(status):"rcx","r11","memory");__builtin_unreachable();}')
    if corpus_entry:lines[-1]=lines[-1].replace('vgi_retained_train()', 'vgi_corpus_train()')
    source=output/'full_model.cpp';source.write_text('\n'.join(lines)+'\n')
    commands=compile_model(root,source,output,optimization,graph_library)
    module=output/'model.imasm'
    report={'parameters':trainable,'buffers':buffers,'config':cfg,'parent':str(checkpoint),
            'objective':objective,
            'learning_rate':learning_rate,'optimizer':optimizer,'optimizer_initial_clock':clock,
            'stream_entry':stream,'stream_gradient_clip':gradient_clip,
            'corpus_oracle':corpus_report,'corpus_entry':corpus_entry,'retained_entry':retained,'retained_operation_oracle':retained,'snapshot_records':snapshot_records,
            'retained_protocol':'VGIR2' if retained else None,'retained_settlement':retained_settle,
            'compiler_optimization':optimization,
            'context_length':context_length,'probe_tokens':probe_tokens,'tape_capacity':tape_capacity,
            'process_update':{'bits':bits,'steps':steps,'continuous_reentry_unroll':unroll},
            'parameter_count':param_count,'commands':commands,
            'graph_library':str(graph_library) if graph_library else None,
            'graph_library_sha256':hashlib.sha256(graph_library.read_bytes()).hexdigest() if graph_library else None,
            'graph_vocabulary':str(graph_vocabulary) if graph_vocabulary else None,
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'module_sha256':hashlib.sha256(module.read_bytes()).hexdigest(),
            'gradient_status':'awaiting_independent_verification'}
    (output/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'built':str(output),'parameters':param_count}))



def compile_model(root,source,output,optimization='O2',library=None):
    flags=['-std=c++20','-'+optimization,'-fno-builtin','-fno-stack-protector','-fno-tree-vectorize',
           '-ffp-contract=off','-mno-avx','-mno-fma','-fno-exceptions','-fno-rtti',
           '-fno-threadsafe-statics','-fno-use-cxa-atexit']
    if library:
        names=set(re.findall(r'extern\s+"C"\s+(?:[A-Za-z_][A-Za-z_0-9]*\s+)+?(vgi_[A-Za-z_0-9]+)\s*\(',source.read_text()))
        flags+=['-ffunction-sections','-fdata-sections','-Wl,--gc-sections',*[f'-Wl,-u,{name}' for name in sorted(names)]]
    extra=[str(library)] if library else []
    commands=[['c++',*flags,'-nostdlib','-static','-no-pie','-Wl,-e,_start',str(source),*extra,'-o',str(output/'model')],
              ['c++',*flags,'-shared','-fPIC',str(source),*extra,'-o',str(output/'model.so')]]
    for command in commands:
        result=subprocess.run(command,capture_output=True,text=True)
        if result.returncode:raise RuntimeError(result.stderr)
    command=[str(root/'Vox/target/release/vox'),'imasm',str(output/'model'),'--output',str(output/'model.imasm')]
    subprocess.run(command,capture_output=True,check=True,timeout=60);commands.append(command);return commands

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--bits',type=int,default=3)
    p.add_argument('--steps',type=int,default=1);p.add_argument('--unroll',type=int,default=0)
    p.add_argument('--objective',choices=('carrier','policy'),default='carrier')
    p.add_argument('--learning-rate',type=float,default=.001)
    p.add_argument('--optimizer',choices=('sgd','adam'),default='sgd')
    p.add_argument('--stream',action='store_true');p.add_argument('--context-length',type=int)
    p.add_argument('--gradient-clip',type=float,default=1.)
    p.add_argument('--retained',action='store_true')
    p.add_argument('--corpus-records',type=Path,help='complete source-bound records to lift as an exact retrieval store')
    p.add_argument('--graph-library',type=Path);p.add_argument('--graph-vocabulary',type=Path)
    p.add_argument('--corpus-entry',action='store_true',help='train by retrieving the baked source records')
    p.add_argument('--retained-settle',action='store_true')
    p.add_argument('--snapshot-records',type=int,default=0,help='emit full source/weight capsules every N completed retained records; zero disables')
    p.add_argument('--optimization',choices=('O1','O2'),default='O1')
    a=p.parse_args();export(a.root.resolve(),a.checkpoint.resolve(),a.out.resolve(),a.bits,a.steps,a.unroll,
                         a.objective,a.learning_rate,a.stream,a.context_length,a.gradient_clip,a.retained,a.retained_settle,a.optimization,a.snapshot_records,a.corpus_records.resolve() if a.corpus_records else None,a.corpus_entry,a.graph_library.resolve() if a.graph_library else None,a.graph_vocabulary.resolve() if a.graph_vocabulary else None,a.optimizer)
