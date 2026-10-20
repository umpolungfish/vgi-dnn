#!/usr/bin/env python3
"""Measure nested and feedback graphs through the existing imasm_core oracle."""
import argparse,ctypes,hashlib,json,re,subprocess
from pathlib import Path


def shapes():
    for op in ('AFWD','AREV','CLINK','EVALT','IMSCRIB','EVALF','ENGAGR','IFIX'):
        yield {'family':'chain:'+op,'depth':0,'nodes':['VINIT',op,'TANCH'],'edges':[[0,1],[1,2]],'readout':2}
        for arity in (2,3):
            for depth in range(1,5):
                nodes=['VINIT'];edges=[]
                def add(name):nodes.append(name);return len(nodes)-1
                def fragment(level):
                    if not level:
                        n=add(op);return n,n
                    fork=add('FSPLIT');fuse=add('FFUSE')
                    for arm in range(arity):
                        head,tail=fragment(level-1) if arm==0 else (lambda n:(n,n))(add('AFWD'))
                        edges.extend([[fork,head],[tail,fuse]])
                    return fork,fuse
                head,tail=fragment(depth);sink=add('TANCH');edges.extend([[0,head],[tail,sink]])
                yield {'family':f'nested:{arity}:{depth}:{op}','depth':depth,'nodes':nodes,'edges':edges,'readout':sink}
    for reverse in (False,True):
        branch=[[3,5],[3,4]] if reverse else [[3,4],[3,5]]
        yield {'family':f'feedback:port_order={int(reverse)}','depth':1,
               'nodes':['VINIT','FFUSE','AREV','FSPLIT','AFWD','TANCH'],
               'edges':[[0,1],[1,2],[2,3],*branch,[4,1]],'readout':5}


def run(root,host,probe,out):
    out.mkdir(parents=True,exist_ok=False)
    vocabulary=json.loads(subprocess.check_output([str(host),'--table'],text=True));byname={r['name']:r for r in vocabulary['tokens']}
    lib=ctypes.CDLL(str(probe/'probe.so'));gate=lib.vgi_graph_gate;gate.argtypes=[ctypes.c_long]*5;gate.restype=ctypes.c_long
    graph_path=out/'current_graph.json';transitions=[];vm_checks={};maximum_rounds=0
    for graph in shapes():
        serialized=json.dumps({k:graph[k] for k in ('nodes','edges','readout')},sort_keys=True,separators=(',',':'))
        signature=hashlib.sha256(serialized.encode()).hexdigest()
        partition='test' if graph['depth']==4 else ('validation' if int(signature[:8],16)%10==0 else 'train')
        for seed in range(16):
            graph_path.write_text(json.dumps({**graph,'seed':seed}))
            actual=json.loads(subprocess.check_output([str(host),'--graph',str(graph_path)],text=True));assert actual['executed'] and not actual['grammar']
            values=[0]*len(graph['edges']);node_inputs=[0]*len(graph['nodes'])
            for turn in range(4*max(1,len(values))+4):
                changed=False
                for node,name in enumerate(graph['nodes']):
                    incoming=0
                    for edge,(_,destination) in enumerate(graph['edges']):
                        if destination==node:incoming|=values[edge]
                    node_inputs[node]=incoming
                    outgoing=[edge for edge,(source,_) in enumerate(graph['edges']) if source==node]
                    for slot,edge in enumerate(outgoing):
                        args=(byname[name]['id'],incoming,seed,slot,len(outgoing))
                        value=gate(*args);assert 0<=value<16
                        if values[edge]!=value:values[edge]=value;changed=True
                        if seed in (0,3,15):vm_checks.setdefault(args,value)
                if not changed:break
            else:raise AssertionError('reference fixed point failed')
            maximum_rounds=max(maximum_rounds,turn+1)
            assert actual['node_inputs']==node_inputs and actual['edge_values']==values
            row={'id':signature+':'+str(seed),'family':signature,'split':partition,'graph':graph,'seed_bits':seed,
                 'return_bits':actual['readout'],'node_inputs':node_inputs,'edge_values':values,'rounds':turn+1,
                 'closure':actual['closure'],'circuit_rank':actual['circuit_rank'],
                 'evidence':'executed imasm_core::flow_values with matching compiled ig_flow_gate propagation'}
            transitions.append(row)
    # Same ordered opcode list, different fork-port wiring, measured return.
    feedback=[r for r in transitions if r['graph']['family'].startswith('feedback:') and r['seed_bits']==3]
    assert len(feedback)==2 and feedback[0]['graph']['nodes']==feedback[1]['graph']['nodes']
    assert feedback[0]['return_bits']!=feedback[1]['return_bits']
    invalid={'nodes':['VINIT','AFWD','TANCH','TANCH'],'edges':[[0,1],[1,2],[1,3]],'seed':3,'readout':2}
    graph_path.write_text(json.dumps(invalid));bad=json.loads(subprocess.check_output([str(host),'--graph',str(graph_path)],text=True));assert not bad['executed'] and bad['grammar']
    vm=[]
    for args,expected in sorted(vm_checks.items()):
        command=[str(root/'Vox/target/release/vox'),'run','vgi_graph_gate',str(probe/'probe.imasm'),'--args',','.join(map(str,args))]
        result=subprocess.run(command,capture_output=True,text=True,timeout=30)
        match=re.search(r'\)\s*=\s*(-?\d+)',result.stdout+result.stderr)
        assert result.returncode==0 and match and int(match[1])==expected,(args,result.stdout,result.stderr)
        vm.append({'args':args,'returned':expected})
    for args in ((256,0,0,0,1),(0,16,0,0,1),(0,0,0,0,2),(6,0,0,3,3)):
        assert gate(*args)==4294967295
    (out/'transitions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in transitions))
    (out/'vocabulary.json').write_text(json.dumps(vocabulary,ensure_ascii=False,indent=2)+'\n')
    (out/'vm_checks.json').write_text(json.dumps(vm,indent=2)+'\n')
    source_files=[root/'MoDoT/imasm_core/src'/name for name in ('flow.rs','check.rs','classic.rs','imasm16_3.rs')]
    report={'status':'passed','transitions':len(transitions),'families':len({r['family'] for r in transitions}),
            'split_counts':{p:sum(r['split']==p for r in transitions) for p in ('train','validation','test')},
            'held_out_depth':4,'maximum_fixed_point_rounds':maximum_rounds,'vm_checks':len(vm),
            'feedback_port_order_returns':[r['return_bits'] for r in feedback],
            'invalid_branch_rejected':True,'source_bindings':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
            'probe_module_sha256':hashlib.sha256((probe/'probe.imasm').read_bytes()).hexdigest()}
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','host','probe','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.root.resolve(),a.host.resolve(),a.probe.resolve(),a.out.resolve())
