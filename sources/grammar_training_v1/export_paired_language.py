#!/usr/bin/env python3
"""Lift a paired language update with one shared clipped Adam step."""
import argparse
import hashlib
import json
from pathlib import Path

from export_full_learning import export, compile_model


def build(root, checkpoint, library, vocabulary, records, out, partial=False):
    row = next(r for r in map(json.loads, records.read_text().splitlines())
               if (r.get('presence_mask') == [1,0] if partial else r.get('evidence_bits') == 3))
    mask = row.get('presence_mask',[1,1])
    export(root, checkpoint, out, retained=True, context_length=16, optimization='O2',
           graph_library=library, graph_vocabulary=vocabulary, optimizer='adam')
    cpp = out / 'full_model.cpp'
    body = cpp.read_text()
    body = body[:body.rindex('extern "C" __attribute__((force_align_arg_pointer)) void _start()')]
    marker = 'static Var retained_chunk('
    assert body.count(marker) == 1
    body = body.replace(marker, 'static int language_channel,language_presence;\nstatic int language_presence_mask[2]={' + ','.join(map(str,mask)) + '};\n' + marker, 1)
    marker = 'int width=kind==1?8:kind==3?2:4,base=kind==1?261:kind==3?265:259;'
    assert body.count(marker) == 1
    body = body.replace(marker, 'int width=language_presence?2:kind==1?8:kind==3?2:4,base=language_presence?259:kind==1?261:kind==3?265:259;')
    marker = 'for(int k=0;k<width;k++){\n                Var z='
    assert body.count(marker) == 1
    body = body.replace(marker, 'for(int k=0;k<width;k++){\n                if(language_presence&&!language_presence_mask[k])continue;\n                Var z=')
    body = body.replace('}(*labels)+=width;', '}(*labels)+=language_presence?(language_presence_mask[0]+language_presence_mask[1]):width;', 1)
    marker = 'retained_output(&carried,i,support,refute,kind!=0||retained_capture);'
    assert body.count(marker) == 1
    body = body.replace(marker, 'retained_output(&carried,i,support,refute,kind!=0||retained_capture||language_channel);')
    marker = 'if(kind==0){\n            Var maximum=support[0];'
    assert body.count(marker) == 1
    body = body.replace(marker, 'if(kind==0){\n            if(language_channel)for(int c=0;c<V;c++)support[c]=refute[c];\n            Var maximum=support[0];')
    marker = 'static void optimizer_apply(const double*gradient,double scale){'
    assert body.count(marker) == 1
    body = body.replace(marker, 'static void optimizer_apply_base(const double*gradient,double scale){', 1)
    marker = 'extern "C" double vgi_optimizer_observe'
    body = body.replace(marker, '''static int language_deferred;
static double language_gradient[PARAMETERS],language_weight=.5;
static void optimizer_apply(const double*g,double scale){
    if(language_deferred){for(int i=0;i<PARAMETERS;i++)language_gradient[i]+=language_weight*g[i];}
    else optimizer_apply_base(g,scale);
}
''' + marker, 1)
    active = [int(row[name + '_target'] is not None) for name in ('support','refutation')]
    banks = [row['prefix'] + list((row[name + '_target'] or '').encode()) + [258] for name in ('support', 'refutation')]
    array = lambda values: '{' + ','.join(map(str, values)) + '}'
    body += f'\nstatic unsigned short language_banks[2][{max(map(len,banks))}]={{{array(banks[0])},{array(banks[1])}}};\n'
    body += f'static int language_lengths[2]={array(list(map(len,banks)))};\n#define LANGUAGE_PREFIX {len(row["prefix"])}\n'
    body += f'static int language_enabled[2]={array(active)};\n#define LANGUAGE_BRANCH_WEIGHT {1/max(1,sum(active)):.17g}\n'
    body += '''
static double language_memory[2][2][D],language_losses[2];
extern "C" double vgi_language_memory(long channel,long evidence,long coordinate){
    return language_memory[channel][evidence][coordinate];}
extern "C" double vgi_language_loss(long channel){return language_losses[channel];}
extern "C" long vgi_paired_language_update(){
    vgi_retained_reset();for(int i=0;i<PARAMETERS;i++)language_gradient[i]=0;
    language_deferred=1;language_weight=LANGUAGE_BRANCH_WEIGHT;
    for(int channel=0;channel<2;channel++){
        if(!language_enabled[channel])continue;
        language_channel=channel;
        for(int i=0;i<language_lengths[channel];i++)retained_bank[i]=language_banks[channel][i];
        long result=vgi_retained_update(LANGUAGE_PREFIX,language_lengths[channel]-LANGUAGE_PREFIX,0,0,1,4);
        if(result<0){language_deferred=language_channel=0;return result;}
        language_losses[channel]=result/1000000.;
        for(int c=0;c<D;c++){language_memory[channel][0][c]=retained_memory_t[c];language_memory[channel][1][c]=retained_memory_f[c];}
    }
    LANGUAGE_PRESENCE_UPDATE
    language_deferred=language_channel=0;double norm2=0;
    for(int i=0;i<PARAMETERS;i++){initial_gradient[i]=language_gradient[i];norm2+=language_gradient[i]*language_gradient[i];}
    double norm=raw_sq(norm2),scale=(STREAM_CLIP>0&&norm>STREAM_CLIP)?STREAM_CLIP/norm:1;
    optimizer_apply_base(language_gradient,scale);return 0;
}
extern "C" __attribute__((force_align_arg_pointer)) void _start(){
    long status=vgi_paired_language_update()<0?2:0;
    if(!status&&vgi_write_current()<0)status=3;
    __asm__ volatile("syscall"::"a"(60L),"D"(status):"rcx","r11","memory");__builtin_unreachable();}
'''
    presence = '''language_channel=0;language_presence=1;language_weight=1;
    for(int i=0;i<LANGUAGE_PREFIX;i++)retained_bank[i]=language_banks[0][i];
    long presence_result=vgi_retained_update(LANGUAGE_PREFIX,0,2,LANGUAGE_EVIDENCE_BITS,1,4);
    language_presence=0;if(presence_result<0){language_deferred=0;return presence_result;}
''' if row.get('evidence_presence') else ''
    body = body.replace('LANGUAGE_PRESENCE_UPDATE', presence.replace('LANGUAGE_EVIDENCE_BITS',str(row['evidence_bits'])))
    cpp.write_text(body)
    metadata = json.loads((out / 'build.json').read_text())
    metadata['paired_language'] = {'record': row, 'records_sha256': hashlib.sha256(records.read_bytes()).hexdigest(),
                                   'loss': 'mean of independent affirmative and refuting continuation mean token losses'}
    metadata['paired_compile_commands'] = compile_model(root, cpp, out, 'O2', library)
    metadata.update(source_sha256=hashlib.sha256(cpp.read_bytes()).hexdigest(),
                    module_sha256=hashlib.sha256((out / 'model.imasm').read_bytes()).hexdigest())
    (out / 'build.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'checkpoint', 'library', 'vocabulary', 'records', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--partial',action='store_true',help='compile one attested corpus continuation with unobserved refutation')
    args = parser.parse_args()
    build(*(getattr(args, name).resolve() for name in ('root','checkpoint','library','vocabulary','records','out')),partial=args.partial)
