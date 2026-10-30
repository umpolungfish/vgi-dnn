#!/usr/bin/env python3
"""Measure unforced learned propagation over complete reference graph families."""
import argparse,ctypes,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path


def evaluate(runtime,dataset,split=None):
    vocabulary=json.loads((dataset/'vocabulary.json').read_text())
    ids={r['name']:r['id'] for r in vocabulary['tokens']}
    lib=ctypes.CDLL(str(runtime/'model.so'));lib.vgi_retained_reset()
    for name,count in [('vgi_graph_reset',3),('vgi_graph_node',2),('vgi_graph_edge',5),('vgi_graph_run',5),('vgi_graph_observe',2)]:
        getattr(lib,name).argtypes=[ctypes.c_long]*count;getattr(lib,name).restype=ctypes.c_long
    totals=defaultdict(Counter);predictions=[]
    for line in (dataset/'transitions.jsonl').open():
        row=json.loads(line)
        if split is not None and row['split']!=split:continue
        graph=row['graph'];n,e=len(graph['nodes']),len(graph['edges'])
        assert lib.vgi_graph_reset(n,e,graph['readout'])==0
        for i,name in enumerate(graph['nodes']):assert lib.vgi_graph_node(i,ids[name])==0
        incoming={};outgoing={}
        for i,(a,b) in enumerate(graph['edges']):
            assert lib.vgi_graph_edge(i,a,b,outgoing.get(a,0),incoming.get(b,0))==0
            outgoing[a]=outgoing.get(a,0)+1;incoming[b]=incoming.get(b,0)+1
        returned=lib.vgi_graph_run(row['seed_bits'],0,0,0,0);assert returned>=0
        edges=[lib.vgi_graph_observe(2,i) for i in range(e)]
        nodes=[lib.vgi_graph_observe(1,i) for i in range(n)]
        stable=lib.vgi_graph_observe(0,0);passes=lib.vgi_graph_observe(0,1)
        result={'id':row['id'],'split':row['split'],'family':graph['family'],'seed_bits':row['seed_bits'],
                'expected':row['return_bits'],'returned':returned,'stable':bool(stable),'passes':passes,
                'edge_values':edges,'node_inputs':nodes,'complete_flow_exact':edges==row['edge_values'] and nodes==row['node_inputs']}
        predictions.append(result)
        for group in ('all',row['split'],'seed:'+str(row['seed_bits'])):
            count=totals[group];count['graphs']+=1;count['return_exact']+=returned==row['return_bits']
            count['complete_flow_exact']+=result['complete_flow_exact'];count['stable']+=stable;count['passes']+=passes
    report={'mode':0,'correction_steps':0,'groups':dict(totals),'selected_split':split,
            'graph_sha256':hashlib.sha256((dataset/'transitions.jsonl').read_bytes()).hexdigest(),
            'module_sha256':hashlib.sha256((runtime/'model.imasm').read_bytes()).hexdigest(),
            'parent':json.loads((runtime/'build.json').read_text())['parent']}
    suffix='_'+split if split is not None else ''
    (runtime/('graph_rollout_predictions'+suffix+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in predictions))
    (runtime/('graph_rollout_evaluation'+suffix+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('runtime','dataset'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--split',choices=('train','validation','test'),help='evaluate only this recorded graph partition')
    a=p.parse_args();evaluate(a.runtime.resolve(),a.dataset.resolve(),a.split)
