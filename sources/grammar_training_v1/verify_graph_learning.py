#!/usr/bin/env python3
"""Check shared neural graph flow, oracle correction, return and retained re-entry."""
import argparse,ctypes,hashlib,json,re,subprocess,sys
from pathlib import Path
import torch
from safetensors.torch import load_file
from graph_format import packet,decode
from retained_conditioning import evaluate
from optimizer_state import load_model,split_state
from torch_reference_optimizer import make_optimizer


def verify(root,runtime,dataset,vm=False):
    torch.set_num_threads(1);sys.path.insert(0,str(root/'bob3ect/reentry_dnn'))
    from dtransformer_torch import DialetheicTransformerTorch
    build=json.loads((runtime/'build.json').read_text());parent=Path(build['parent']);initial=load_model(parent/'model.safetensors')
    vocabulary=json.loads((dataset/'vocabulary.json').read_text());ids={r['name']:r['id'] for r in vocabulary['tokens']}
    rows=[json.loads(l) for l in (dataset/'transitions.jsonl').open()]
    lib=ctypes.CDLL(str(runtime/'model.so'))
    for name,count in [('vgi_graph_reset',3),('vgi_graph_node',2),('vgi_graph_edge',5),('vgi_graph_run',5),
                       ('vgi_graph_continue',4),('vgi_graph_observe',2)]:
        getattr(lib,name).argtypes=[ctypes.c_long]*count;getattr(lib,name).restype=ctypes.c_long
    lib.ig_flow_gate.argtypes=[ctypes.c_uint]*5;lib.ig_flow_gate.restype=ctypes.c_uint
    lib.vgi_full_parameter.argtypes=[ctypes.c_long];lib.vgi_full_parameter.restype=ctypes.c_double
    def setup(row):
        graph=row['graph'];assert lib.vgi_graph_reset(len(graph['nodes']),len(graph['edges']),graph['readout'])==0
        for i,name in enumerate(graph['nodes']):assert lib.vgi_graph_node(i,ids[name])==0
        ins={};outs={}
        for i,(a,b) in enumerate(graph['edges']):
            assert lib.vgi_graph_edge(i,a,b,outs.get(a,0),ins.get(b,0))==0
            outs[a]=outs.get(a,0)+1;ins[b]=ins.get(b,0)+1
    def new_model():
        model=DialetheicTransformerTorch(**build['config']).double();model.load_state_dict(initial);return model
    def simulation(model,row,mode,steps):
        g=row['graph'];values=[0]*len(g['edges']);node_values=[0]*len(g['nodes']);before=values.copy();after=values.copy();expected=values.copy()
        errors=post_errors=updates=calls=0
        optimizer=make_optimizer(model,build)
        for iteration in range(4*max(1,len(values))+4):
            changed=False;last_updates=0
            for node,name in enumerate(g['nodes']):
                incoming=0
                for e,(_,target) in enumerate(g['edges']):
                    if target==node:incoming|=values[e]
                node_values[node]=incoming;outgoing=[e for e,(source,_) in enumerate(g['edges']) if source==node]
                for slot,e in enumerate(outgoing):
                    correct=lib.ig_flow_gate(ids[name],incoming,row['seed_bits'],slot,len(outgoing));assert correct<16
                    query=[256,71,ids[name],266+incoming,266+row['seed_bits'],len(outgoing),slot,257]
                    def propose():
                        with torch.no_grad():result=evaluate(model,query,[],context=16,kind=2,bits=correct)
                        return sum(int(result['first_logits'][k%2][259+k//2]>0)<<k for k in range(4))
                    first=propose();last=first
                    if mode in (2,3) and first!=correct and steps:
                        for _ in range(steps):
                            optimizer.zero_grad();result=evaluate(model,query,[],context=16,kind=2,bits=correct)
                            for loss in result['loss_windows']:loss.backward()
                            for p in model.parameters():p.grad.div_(result['labels'])
                            norm=float(torch.sqrt(sum((p.grad*p.grad).sum() for p in model.parameters())))
                            scale=min(1.,build['stream_gradient_clip']/norm) if norm else 1.
                            for p in model.parameters():p.grad.mul_(scale)
                            optimizer.step()
                        updates+=steps;last_updates+=steps;last=propose()
                    before[e]=first;after[e]=last;expected[e]=correct;errors+=first!=correct;post_errors+=last!=correct;calls+=1
                    value=correct if mode==2 else last
                    if values[e]!=value:values[e]=value;changed=True
            if not changed:break
        return {'values':values,'node_inputs':node_values,'before':before,'after':after,'expected':expected,
                'passes':iteration+1,'stable':not changed,'calls':calls,'errors':errors,'post_errors':post_errors,'updates':updates}
    # Exact source oracle over every reference graph, including held-out depth.
    for row in rows:
        setup(row);value=lib.vgi_graph_run(row['seed_bits'],1,0,0,0)
        assert value==row['return_bits'] and lib.vgi_graph_observe(0,0)==1
        assert [lib.vgi_graph_observe(2,i) for i in range(len(row['edge_values']))]==row['edge_values']
        assert [lib.vgi_graph_observe(1,i) for i in range(len(row['node_inputs']))]==row['node_inputs']
    continuation=[]
    for row in [r for r in rows if (r['graph']['depth']==4 or r['graph']['family'].startswith('feedback:')) and r['seed_bits']==3]:
        setup(row);lib.vgi_graph_run(3,1,1,0,0)
        first=[lib.vgi_graph_observe(2,i) for i in range(len(row['edge_values']))]
        assert lib.vgi_graph_continue(1,0,0,0)==row['return_bits']
        assert [lib.vgi_graph_observe(2,i) for i in range(len(row['edge_values']))]==row['edge_values']
        continuation.append({'id':row['id'],'initial_pass_edges':first,'total_passes':lib.vgi_graph_observe(0,12)})
    free=[]
    selected=[rows[i] for i in (0,3,15,100,500,900,len(rows)-2,len(rows)-1)]
    for row in selected:
        model=new_model();reference=simulation(model,row,0,0);lib.vgi_retained_reset();setup(row)
        returned=lib.vgi_graph_run(row['seed_bits'],0,0,0,0)
        assert returned==reference['node_inputs'][row['graph']['readout']]
        for key,kind in [('values',2),('before',3),('after',4),('expected',5)]:
            assert [lib.vgi_graph_observe(kind,i) for i in range(len(row['edge_values']))]==reference[key]
        assert lib.vgi_graph_observe(0,0)==int(reference['stable'])
        free.append({'id':row['id'],'returned':returned,'oracle_return':row['return_bits'],'stable':reference['stable'],'passes':reference['passes']})
    # Update and correction on the same graph and shared weights.
    row=next(r for r in rows if r['graph']['family']=='nested:3:2:AREV' and r['seed_bits']==3)
    model=new_model();reference=simulation(model,row,2,1);lib.vgi_retained_reset();setup(row)
    assert lib.vgi_graph_run(3,2,0,1,0)==row['return_bits']
    assert lib.vgi_graph_observe(0,5)==reference['updates']
    position=0;error=0.;parameters=dict(model.named_parameters())
    for entry in build['parameters']:
        actual=torch.tensor([lib.vgi_full_parameter(i) for i in range(position,position+entry['size'])],dtype=torch.double)
        error=max(error,float((actual-parameters[entry['key']].detach().flatten()).abs().max()));position+=entry['size']
    assert error<1e-8
    for key,kind in [('values',2),('before',3),('after',4),('expected',5)]:
        assert [lib.vgi_graph_observe(kind,i) for i in range(len(row['edge_values']))]==reference[key]
    payload=packet(row['graph'],3,vocabulary);(runtime/'checked_graph.input').write_bytes(payload)
    process=subprocess.run([str(runtime/'model')],input=payload,capture_output=True,timeout=90)
    assert process.returncode==0,process.stderr
    (runtime/'checked_graph.stdout').write_bytes(process.stdout)
    weights,retained,returned=decode(process.stdout)
    checkpoint=runtime/'checked_graph_return';checkpoint.mkdir(exist_ok=True)
    (checkpoint/'model.safetensors').write_bytes(weights);actual=load_model(checkpoint/'model.safetensors')
    assert max(float((actual[k].double()-model.state_dict()[k].double()).abs().max()) for k in actual)<1e-8
    assert returned['values']==reference['values'] and returned['opcodes']==[ids[n] for n in row['graph']['nodes']]
    (checkpoint/'graph_state.json').write_text(json.dumps(returned,indent=2)+'\n')
    (checkpoint/'retained_state.json').write_text(json.dumps(retained,indent=2)+'\n')
    report={'status':'passed','oracle_graphs':len(rows),'retained_continuations':len(continuation),'continuation_checks':continuation,
            'free_neural_graphs':free,'checked_graph':{'id':row['id'],'updates':reference['updates'],'parameter_max_error':error,
            'proposals_preserved':True,'source_and_carrier_returned':True},'module_sha256':hashlib.sha256((runtime/'model.imasm').read_bytes()).hexdigest()}
    (runtime/'graph_learning_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    if vm:
        # The complete ported feedback graph runs through the lifted entry.
        row=next(r for r in rows if r['graph']['family']=='feedback:port_order=0' and r['seed_bits']==3)
        command=[str(root/'Vox/target/release/vox'),'run','vgi_graph_stream',str(runtime/'model.imasm'),'--args','1,0,0,1']
        process=subprocess.run(command,input=packet(row['graph'],3,vocabulary),capture_output=True,timeout=180)
        observation=(process.stdout+process.stderr).decode();(runtime/'graph_vm.log').write_text(observation)
        match=re.search(r'\)\s*=\s*(-?\d+)',observation);assert process.returncode==0 and match and int(match[1])==1|(row['return_bits']<<1),observation
        report['vm']={'command':command,'id':row['id'],'packed_return':int(match[1]),'passed':True}
        (runtime/'graph_learning_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':'passed','oracle_graphs':len(rows),'free_neural_graphs':len(free),'checked_updates':reference['updates'],
                      'parameter_error':error,'vm':report.get('vm')}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','runtime','dataset'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--vm',action='store_true');a=p.parse_args();verify(a.root.resolve(),a.runtime.resolve(),a.dataset.resolve(),a.vm)
