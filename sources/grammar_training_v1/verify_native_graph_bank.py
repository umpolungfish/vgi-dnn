#!/usr/bin/env python3
"""Prepare a saved source-bound graph bank inside a lifted native executable."""
import argparse
import ctypes
import json
from pathlib import Path
import sys
import torch
from composed_graph_reference import forward
from export_full_learning import compile_model
from graph_reentry_bank import load,initial
from optimizer_state import load_model
from train_cuda_trajectories import sha


def verify(root,runtime,dataset,out):
    torch.set_num_threads(1);sys.path.insert(0,str(root/'bob3ect/reentry_dnn'))
    from dtransformer_torch import DialetheicTransformerTorch
    build=json.loads((runtime/'build.json').read_text());parent=Path(build['parent'])
    row=json.loads((runtime/'training_graphs.jsonl').read_text().splitlines()[0])
    rows=[json.loads(s) for s in (dataset/'transitions.jsonl').open()]
    vocabulary=json.loads((dataset/'vocabulary.json').read_text());ids={r['name']:r['id'] for r in vocabulary['tokens']}
    bank=load(parent,rows,sha(dataset/'transitions.jsonl'),sha(dataset/'vocabulary.json'),build['config']['d_model'],build['optimizer_initial_clock'])
    state=initial(bank,row,build['config']['d_model'],build['optimizer_initial_clock']);assert state is not None
    graph=row['graph'];out.mkdir(parents=True,exist_ok=False)
    body=(runtime/'full_model.cpp').read_text();body=body[:body.rindex('extern "C" __attribute__((force_align_arg_pointer)) void _start()')]
    def array(value):
        return '{'+','.join(array(v) if isinstance(v,list) else format(v,'.17g') for v in value)+'}'
    body+='\nstatic const double bank_t[][D]='+array(state['memory_t'])+';\n'
    body+='static const double bank_f[][D]='+array(state['memory_f'])+';\n'
    body+='static const double bank_p[][4]='+array(state['probabilities'])+';\n'
    calls=['vgi_retained_reset();',f'if(vgi_graph_reset({len(graph["nodes"])},{len(graph["edges"])},{graph["readout"]})<0)return -1;']
    for i,name in enumerate(graph['nodes']):calls.append(f'if(vgi_graph_node({i},{ids[name]})<0)return -2;')
    outgoing={};incoming={}
    for i,(a,b) in enumerate(graph['edges']):
        calls.append(f'if(vgi_graph_edge({i},{a},{b},{outgoing.get(a,0)},{incoming.get(b,0)})<0)return -3;')
        outgoing[a]=outgoing.get(a,0)+1;incoming[b]=incoming.get(b,0)+1
    calls.append(f'long source=vgi_graph_run({row["seed_bits"]},1,0,0,0);if(source<0)return -4;')
    calls.append('composed_source_return=source;composed_total_passes=0;')
    calls.append('for(long i=0;i<graph_edges;i++){composed_source_edges[i]=graph_value[i];for(int c=0;c<D;c++){composed_memory[i][0][c]=bank_t[i][c];composed_memory[i][1][c]=bank_f[i][c];}for(int c=0;c<4;c++)composed_probability[i][c]=bank_p[i][c];}')
    calls.append('for(long i=0;i<graph_nodes;i++)composed_source_nodes[i]=graph_in[i];for(int i=0;i<14;i++)composed_source_diagnostics[i]=graph_diagnostics[i];')
    calls.append('composed_binding=composed_graph_binding();composed_started=1;return 0;')
    body+='extern "C" long vgi_banked_prepare(){'+''.join(calls)+'}\n'
    body+='''extern "C" __attribute__((force_align_arg_pointer)) void _start(){
long status=vgi_banked_prepare()<0?2:0;if(!status&&vgi_composed_run(2,1,250000,0,0)<0)status=3;
if(!status&&vgi_composed_save()<0)status=4;
__asm__ volatile("syscall"::"a"(60L),"D"(status):"rcx","r11","memory");__builtin_unreachable();}\n'''
    cpp=out/'full_model.cpp';cpp.write_text(body)
    commands=compile_model(root,cpp,out,'O2',Path(build['graph_library']))
    native=ctypes.CDLL(str(out/'model.so'))
    native.vgi_composed_run.argtypes=[ctypes.c_long]*5;native.vgi_composed_run.restype=ctypes.c_long
    native.vgi_composed_observe.argtypes=[ctypes.c_long]*4;native.vgi_composed_observe.restype=ctypes.c_double
    native.vgi_full_gradient.argtypes=[ctypes.c_long];native.vgi_full_gradient.restype=ctypes.c_double
    native.ig_flow_gate.argtypes=[ctypes.c_uint]*5;native.ig_flow_gate.restype=ctypes.c_uint
    model=DialetheicTransformerTorch(**build['config']).double();model.load_state_dict(load_model(parent/'model.safetensors'))
    result=forward(model,row,vocabulary,native.ig_flow_gate,2,.25,initial=state)
    result['loss'].backward();named=dict(model.named_parameters())
    expected=torch.cat([named[p['key']].grad.flatten() for p in build['parameters']])
    assert native.vgi_banked_prepare()==0
    assert native.vgi_composed_run(2,0,250000,0,0)>=0
    loss_error=abs(float(result['loss'].detach())-native.vgi_composed_observe(0,0,0,0))
    memory_error=0.
    for e,pair in enumerate(result['memory']):
        for channel,value in enumerate(pair):
            actual=torch.tensor([native.vgi_composed_observe(2,e,channel,c) for c in range(model.d)],dtype=torch.double)
            memory_error=max(memory_error,float((actual-value.detach().flatten()).abs().max()))
    assert native.vgi_banked_prepare()==0
    assert native.vgi_composed_run(2,1,250000,0,0)>=0
    actual=torch.tensor([native.vgi_full_gradient(i) for i in range(len(expected))],dtype=torch.double)
    gradient_error=float((actual-expected).abs().max())
    with torch.no_grad():reset=forward(model,row,vocabulary,native.ig_flow_gate,2,.25)
    reset_difference=abs(float(reset['loss'])-float(result['loss'].detach()))
    assert max(loss_error,memory_error,gradient_error)<1e-8
    assert reset_difference>1e-10
    report={'passed':True,'parent':str(parent),'parent_sha256':sha(parent/'model.safetensors'),'graph':row,
            'bank_state':state,'module_sha256':sha(out/'model.imasm'),'source_sha256':sha(cpp),
            'loss_error':loss_error,'memory_error':memory_error,'gradient_error':gradient_error,
            'reset_loss_difference':reset_difference,'commands':commands,
            'scope':'saved graph bank baked into native executing register; graph-only composed objective and full parameter gradient versus CPU; shared text/typed objective checked separately'}
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('bank_state','graph','commands')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','runtime','dataset','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();verify(a.root.resolve(),a.runtime.resolve(),a.dataset.resolve(),a.out.resolve())
