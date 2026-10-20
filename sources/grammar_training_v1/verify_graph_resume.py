#!/usr/bin/env python3
"""Compare graph checkpoint re-entry with uninterrupted shared-weight learning."""
import argparse,ctypes,hashlib,json,os,shutil,subprocess
from pathlib import Path
from graph_format import decode,resume_packet
from export_graph_learning import build
import torch
from safetensors.torch import load


def verify(root,runtime,dataset):
    report=json.loads((runtime/'build.json').read_text())
    vocabulary=json.loads((dataset/'vocabulary.json').read_text())
    ids={r['name']:r['id'] for r in vocabulary['tokens']}
    rows=[json.loads(line) for line in (dataset/'transitions.jsonl').open()]
    row=next(r for r in rows if r['graph']['family']=='nested:3:2:AREV' and r['seed_bits']==3)
    graph=row['graph'];lib=ctypes.CDLL(str(runtime/'model.so'))
    for name,count in [('vgi_graph_reset',3),('vgi_graph_node',2),('vgi_graph_edge',5),
                       ('vgi_graph_run',5),('vgi_graph_continue',4)]:
        getattr(lib,name).argtypes=[ctypes.c_long]*count;getattr(lib,name).restype=ctypes.c_long
    lib.vgi_retained_reset();assert lib.vgi_graph_reset(len(graph['nodes']),len(graph['edges']),graph['readout'])==0
    for i,name in enumerate(graph['nodes']):assert lib.vgi_graph_node(i,ids[name])==0
    incoming={};outgoing={}
    for i,(a,b) in enumerate(graph['edges']):
        assert lib.vgi_graph_edge(i,a,b,outgoing.get(a,0),incoming.get(b,0))==0
        outgoing[a]=outgoing.get(a,0)+1;incoming[b]=incoming.get(b,0)+1
    def save(name):
        path=runtime/name
        with path.open('wb') as stream:
            original=os.dup(1)
            try:
                os.dup2(stream.fileno(),1);result=lib.vgi_graph_save()
            finally:os.dup2(original,1);os.close(original)
        assert result==0
        return decode(path.read_bytes())
    assert lib.vgi_graph_run(3,2,1,1,0)>=0
    weights,retained,partial=save('partial_graph.stdout')
    assert partial['diagnostics'][5]>0 and partial['diagnostics'][12]==1
    assert lib.vgi_graph_continue(2,0,1,0)==row['return_bits']
    expected_weights,expected_retained,expected_graph=save('continued_graph.stdout')
    checkpoint=runtime/'partial_checkpoint';checkpoint.mkdir(exist_ok=True)
    (checkpoint/'model.safetensors').write_bytes(weights)
    parent=Path(report['parent'])
    for name in ('config.json','tokenizer.json','execution_input.json'):
        if (parent/name).exists():shutil.copy2(parent/name,checkpoint/name)
    resumed=runtime/f'resumed_runtime_{len(list(runtime.glob("resumed_runtime*"))):03d}'
    build(root,checkpoint,resumed,Path(report['graph_learning']['library']),Path(report['graph_learning']['vocabulary']),2,1,0,report.get('optimizer','sgd'))
    payload=resume_packet(partial,retained);(resumed/'resume.input').write_bytes(payload)
    result=subprocess.run([str(resumed/'model')],input=payload,capture_output=True,timeout=90)
    assert result.returncode==0,result.stderr
    (resumed/'resume.stdout').write_bytes(result.stdout)
    actual_weights,actual_retained,actual_graph=decode(result.stdout)
    actual_tensors,expected_tensors=load(actual_weights),load(expected_weights)
    assert actual_tensors.keys()==expected_tensors.keys()
    assert all(torch.equal(actual_tensors[k],expected_tensors[k]) for k in actual_tensors),'updated weights differ after checkpoint re-entry'
    assert actual_retained==expected_retained,'retained source or memory differs after checkpoint re-entry'
    assert actual_graph==expected_graph,'ported carrier or continuation differs after checkpoint re-entry'
    receipt={'status':'passed','graph_id':row['id'],'partial_updates':partial['diagnostics'][5],
             'resumed_updates':actual_graph['diagnostics'][5],'total_passes':actual_graph['diagnostics'][12],
             'returned_bits':actual_graph['diagnostics'][11],'weight_tensors_exact':True,
             'retained_state_exact':True,'graph_state_exact':True,
             'partial_checkpoint_sha256':hashlib.sha256(weights).hexdigest(),
             'resumed_checkpoint_sha256':hashlib.sha256(actual_weights).hexdigest()}
    (runtime/'graph_resume_verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','runtime','dataset'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();verify(a.root.resolve(),a.runtime.resolve(),a.dataset.resolve())
