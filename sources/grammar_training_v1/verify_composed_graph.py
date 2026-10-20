#!/usr/bin/env python3
"""Compare composed graph values, gradients, Adam state and upstream memory pullbacks."""
import argparse
import ctypes
import hashlib
import json
import sys
from pathlib import Path
import torch
from optimizer_state import load_model
from torch_reference_optimizer import make_optimizer
from composed_graph_reference import forward


def verify(root,runtime,dataset):
    torch.set_num_threads(1);sys.path.insert(0,str(root/'bob3ect/reentry_dnn'))
    from dtransformer_torch import DialetheicTransformerTorch
    build=json.loads((runtime/'build.json').read_text());parent=Path(build['parent'])
    rows=[json.loads(line) for line in (dataset/'transitions.jsonl').open()]
    vocabulary=json.loads((dataset/'vocabulary.json').read_text());ids={r['name']:r['id'] for r in vocabulary['tokens']}
    lib=ctypes.CDLL(str(runtime/'model.so'))
    for name,n in [('vgi_graph_reset',3),('vgi_graph_node',2),('vgi_graph_edge',5),('vgi_graph_run',5),('vgi_composed_run',5),('vgi_graph_observe',2)]:
        getattr(lib,name).argtypes=[ctypes.c_long]*n;getattr(lib,name).restype=ctypes.c_long
    for name,n in [('vgi_composed_observe',4),('vgi_composed_adjoint',3),('vgi_composed_trace_observe',4),('vgi_full_parameter',1),('vgi_full_gradient',1),('vgi_optimizer_observe',2)]:
        getattr(lib,name).argtypes=[ctypes.c_long]*n;getattr(lib,name).restype=ctypes.c_double
    lib.ig_flow_gate.argtypes=[ctypes.c_uint]*5;lib.ig_flow_gate.restype=ctypes.c_uint
    def setup(row):
        lib.vgi_retained_reset();g=row['graph']
        assert lib.vgi_graph_reset(len(g['nodes']),len(g['edges']),g['readout'])==0
        for i,name in enumerate(g['nodes']):assert lib.vgi_graph_node(i,ids[name])==0
        outgoing={};incoming={}
        for i,(a,b) in enumerate(g['edges']):
            assert lib.vgi_graph_edge(i,a,b,outgoing.get(a,0),incoming.get(b,0))==0
            outgoing[a]=outgoing.get(a,0)+1;incoming[b]=incoming.get(b,0)+1
        assert lib.vgi_graph_run(row['seed_bits'],1,0,0,0)==row['return_bits']
        assert [lib.vgi_graph_observe(2,i) for i in range(len(g['edges']))]==row['edge_values']
    selected=[next(row for row in rows if row['split']=='train' and row['graph']['family']==family and row['seed_bits']==seed)
              for family,seed in [('chain:AREV',1),('nested:2:1:AREV',3),('feedback:port_order=0',3)]]
    reports=[]
    for row in selected:
        controls=[];gradients=[]
        for detach in (False,True):
            setup(row);model=DialetheicTransformerTorch(**build['config']).double()
            model.load_state_dict(load_model(parent/'model.safetensors'));optimizer=make_optimizer(model,build);optimizer.zero_grad()
            result=forward(model,row,vocabulary,lib.ig_flow_gate,2,0.,detach)
            for mt,mf in result['memory']:mt.retain_grad();mf.retain_grad()
            for event in result['trace']:
                for value in (*event['memory'],event['probability']):value.retain_grad()
            before=float(result['loss'].detach())
            assert lib.vgi_composed_run(2,0,0,int(detach),1)>=0
            native_before=lib.vgi_composed_observe(0,0,0,0)
            before_error=abs(before-native_before);assert before_error<1e-8,(before,native_before)
            result['loss'].backward();expected_gradients={name:p.grad.detach().clone() for name,p in model.named_parameters()}
            temporal=[]
            for event in result['trace']:
                temporal.append({'info':[event['pass'],event['edge'],event['input'],event['target']],
                    'memory':[v.detach().flatten().clone() for v in event['memory']],
                    'gradient':[v.grad.detach().flatten().clone() if v.grad is not None else torch.zeros(model.d,dtype=torch.double) for v in event['memory']],
                    'probability':event['probability'].detach().clone(),
                    'probability_gradient':event['probability'].grad.detach().clone() if event['probability'].grad is not None else torch.zeros(4,dtype=torch.double)})
            memory_gradients=[[v.grad.detach().flatten() if v.grad is not None else torch.zeros(model.d,dtype=torch.double) for v in pair] for pair in result['memory']]
            norm=float(torch.sqrt(sum((p.grad*p.grad).sum() for p in model.parameters())));scale=min(1.,build['stream_gradient_clip']/norm) if norm else 1.
            for p in model.parameters():p.grad.mul_(scale)
            optimizer.step();setup(row)
            code=lib.vgi_composed_run(2,1,0,int(detach),1)
            assert code>=0,{'code':code,'peak':lib.vgi_composed_observe(0,0,0,4),'graph':row['id']}
            expected=dict(model.named_parameters());position=0;parameter_error=gradient_error=moment_error=adjoint_error=0.;actual_gradient=[]
            for entry in build['parameters']:
                key=entry['key'];p=expected[key];size=entry['size']
                actual=torch.tensor([lib.vgi_full_parameter(i) for i in range(position,position+size)],dtype=torch.double)
                gradient=torch.tensor([lib.vgi_full_gradient(i) for i in range(position,position+size)],dtype=torch.double)
                parameter_error=max(parameter_error,float((actual-p.detach().flatten()).abs().max()))
                gradient_error=max(gradient_error,float((gradient-expected_gradients[key].flatten()).abs().max()));actual_gradient.extend(gradient.tolist())
                for kind,name in ((1,'exp_avg'),(2,'exp_avg_sq')):
                    actual=torch.tensor([lib.vgi_optimizer_observe(kind,i) for i in range(position,position+size)],dtype=torch.double)
                    moment_error=max(moment_error,float((actual-optimizer.state[p][name].flatten()).abs().max()))
                position+=size
            native_adjoint=[];upstream_adjoint=[]
            for edge,(a,b) in enumerate(row['graph']['edges']):
                for channel in (0,1):
                    actual=torch.tensor([lib.vgi_composed_adjoint(edge,channel,i) for i in range(model.d)],dtype=torch.double)
                    adjoint_error=max(adjoint_error,float((actual-memory_gradients[edge][channel]).abs().max()))
                    if edge==0:native_adjoint.extend(actual.tolist())
                    if b!=row['graph']['readout']:upstream_adjoint.extend(actual.tolist())
            temporal_error=0.;temporal_upstream=[]
            assert int(lib.vgi_composed_trace_observe(0,0,0,0))==len(temporal)
            for i,event in enumerate(temporal):
                assert [int(lib.vgi_composed_trace_observe(1,i,0,k)) for k in range(4)]==event['info']
                for channel in (0,1):
                    for kind,key in ((2,'memory'),(3,'gradient')):
                        actual=torch.tensor([lib.vgi_composed_trace_observe(kind,i,channel,c) for c in range(model.d)],dtype=torch.double)
                        temporal_error=max(temporal_error,float((actual-event[key][channel]).abs().max()))
                        if kind==3 and row['graph']['edges'][event['info'][1]][1]!=row['graph']['readout']:temporal_upstream.extend(actual.tolist())
                for kind,key in ((4,'probability'),(5,'probability_gradient')):
                    actual=torch.tensor([lib.vgi_composed_trace_observe(kind,i,0,k) for k in range(4)],dtype=torch.double)
                    temporal_error=max(temporal_error,float((actual-event[key]).abs().max()))
            with torch.no_grad():after=forward(model,row,vocabulary,lib.ig_flow_gate,2,0.,detach)
            loss_error=abs(lib.vgi_composed_observe(0,0,0,0)-float(after['loss']))
            state_error=0.
            for edge in range(len(row['graph']['edges'])):
                for bit in range(4):state_error=max(state_error,abs(lib.vgi_composed_observe(1,edge,0,bit)-float(after['probabilities'][edge][bit])))
                for channel in (0,1):
                    values=after['memory'][edge][channel].flatten()
                    for c in range(model.d):state_error=max(state_error,abs(lib.vgi_composed_observe(2,edge,channel,c)-float(values[c])))
            assert max(parameter_error,gradient_error,moment_error,adjoint_error,loss_error,state_error,temporal_error)<1e-8,(row['graph']['family'],parameter_error,gradient_error,moment_error,adjoint_error,loss_error,state_error,temporal_error)
            assert lib.vgi_optimizer_observe(0,0)==build['optimizer_initial_clock']+1
            controls.append({'detach_incoming_memory':detach,'before_loss':before,'gradient_error':gradient_error,
                'parameter_error':parameter_error,'moment_error':moment_error,'adjoint_error':adjoint_error,
                'edge0_adjoint_norm':sum(v*v for v in native_adjoint)**.5,'state_error':state_error,
                'upstream_adjoint_norm':sum(v*v for v in upstream_adjoint)**.5,
                'temporal_state_and_adjoint_error':temporal_error,'temporal_upstream_adjoint_norm':sum(v*v for v in temporal_upstream)**.5,
                'peak_tape_nodes':int(lib.vgi_composed_observe(0,0,0,4)),'returned_bits':code})
            gradients.append(torch.tensor(actual_gradient,dtype=torch.double))
        assert abs(controls[0]['before_loss']-controls[1]['before_loss'])<1e-12
        difference=float((gradients[0]-gradients[1]).abs().max());assert difference>1e-10
        assert controls[0]['temporal_upstream_adjoint_norm']>1e-10,(row['graph']['family'],controls)
        assert controls[1]['temporal_upstream_adjoint_norm']<1e-12,(row['graph']['family'],controls)
        reports.append({'graph_id':row['id'],'family':row['graph']['family'],'seed':row['seed_bits'],
                        'terminal_only_loss':True,'gradient_change_on_detach':difference,'controls':controls})
    report={'status':'passed','graphs':reports,'module_sha256':hashlib.sha256((runtime/'model.imasm').read_bytes()).hexdigest(),
            'scope':'two-pass composed forward and full-network Adam step; terminal loss reaches upstream continuous edge memory; training-only fixtures'}
    (runtime/'composed_gradient_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','runtime','dataset'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();verify(a.root.resolve(),a.runtime.resolve(),a.dataset.resolve())
