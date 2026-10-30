#!/usr/bin/env python3
"""Recover continuous states only from a checkpoint's selected training lineage."""
import argparse
import ctypes
import json
from pathlib import Path
import shutil
import subprocess
from safetensors.torch import load_file
import graph_reentry_bank as bank_io
from train_cuda_trajectories import sha


def recover(root,checkpoint,dataset,oracle_runtime,out):
    lineage=json.loads((checkpoint/'dataset_lineage.json').read_text())
    entries=lineage['composed_graph_updates']
    rows=[json.loads(s) for s in (dataset/'transitions.jsonl').open()];sources={r['id']:r for r in rows}
    vocabulary=json.loads((dataset/'vocabulary.json').read_text());ids={r['name']:r['id'] for r in vocabulary['tokens']}
    clock=int(load_file(str(checkpoint/'model.safetensors'))['__optimizer.clock'][0])
    width=json.loads((checkpoint/'config.json').read_text())['d_model']
    source_sha=sha(dataset/'transitions.jsonl');vocabulary_sha=sha(dataset/'vocabulary.json')
    bank=bank_io.load(checkpoint,rows,source_sha,vocabulary_sha,width,clock);previous_count=len(bank['states'])
    lib=ctypes.CDLL(str(oracle_runtime/'model.so'));oracle=lib.ig_flow_gate
    oracle.argtypes=[ctypes.c_uint]*5;oracle.restype=ctypes.c_uint
    oracle_build=json.loads((oracle_runtime/'build.json').read_text())
    paths=subprocess.run(['rg','--files',str(root/'VGI'),'-g','trajectory_training_receipt.json'],capture_output=True,text=True,check=True).stdout.splitlines()
    receipts=[(Path(p),json.loads(Path(p).read_text())) for p in sorted(paths)]
    history={e['state_sha256']:e for e in bank.get('recovered_history',[])}
    recovered=[];calls=0
    for stage,e in enumerate(entries):
        if e['source_sha256']!=source_sha:raise ValueError('historical graph corpus changed')
        matches=[(p,r) for p,r in receipts if r['parent_checkpoint_sha256']==e['parent_checkpoint_sha256']
            and [g['id'] for g in r['training_graphs']]==e['training_ids']
            and r['training_graphs'][-1]['state_sha256']==e['carried_state_sha256']]
        if not matches:raise ValueError(f'missing selected stage {stage}')
        receipt,report=matches[0];runtime=receipt.parent
        verification=json.loads((runtime/'cuda_return_verification.json').read_text())
        assert verification['status']=='passed' and verification['checkpoint_sha256']==report['checkpoint_sha256']
        build=json.loads((runtime/'build.json').read_text())
        assert build['graph_library_sha256']==oracle_build['graph_library_sha256']
        for index,record in enumerate(report['training_graphs']):
            path=runtime/f'trajectory_{index:04d}.json';assert sha(path)==record['state_sha256']
            state=json.loads(path.read_text());row=sources[record['id']]
            assert row['split']=='train' and state['graph']==row
            bank_io.validate(state,row,width,clock)
            graph=row['graph'];outgoing={}
            for edge,(a,b) in enumerate(graph['edges']):outgoing.setdefault(a,[]).append(edge)
            for event in state['observations']:
                edge=event['edge'];node=graph['edges'][edge][0];fan=outgoing[node]
                expected=oracle(ids[graph['nodes'][node]],event['input'],row['seed_bits'],fan.index(edge),len(fan))
                assert expected==event['target'],'historical oracle target binding changed'
                assert event['missing']==event['target']&~event['proposal']&15
                assert event['additional']==event['proposal']&~event['target']&15
                calls+=1
            history[record['state_sha256']]={'state_sha256':record['state_sha256'],'state':state,
                'receipt':str(receipt),'receipt_sha256':sha(receipt),'stage_in_selected_lineage':stage}
            old=bank_io.initial(bank,row,width,clock)
            if old is None or state['optimizer_clock']>old['optimizer_clock']:
                bank_io.store(bank,state,width,clock)
        recovered.append({'stage':stage,'receipt':str(receipt),'states':len(report['training_graphs'])})
    bank['recovered_history']=list(history.values())
    bank['retain_reentry_history']=True
    out.mkdir(parents=True,exist_ok=False);returned=out/'returned_checkpoint';shutil.copytree(checkpoint,returned)
    (returned/'composed_graph_bank.json').write_text(json.dumps(bank,indent=2)+'\n')
    event={'parent':str(checkpoint),'model_sha256':sha(checkpoint/'model.safetensors'),
        'source_sha256':source_sha,'vocabulary_sha256':vocabulary_sha,'prior_graphs':previous_count,
        'returned_graphs':len(bank['states']),'historical_states_retained':len(history),
        'oracle_observations_checked':calls,'stages':recovered,'bank_sha256':sha(returned/'composed_graph_bank.json')}
    lineage.setdefault('graph_memory_recoveries',[]).append(event)
    (returned/'dataset_lineage.json').write_text(json.dumps(lineage,indent=2)+'\n')
    for path in checkpoint.iterdir():
        if path.is_file() and path.name not in ('dataset_lineage.json','composed_graph_bank.json'):
            assert path.read_bytes()==(returned/path.name).read_bytes()
    event['scope']='selected-lineage memory recovery; no parameter, Adam or cursor update; older states and disagreements retained in history'
    (out/'verification.json').write_text(json.dumps(event,indent=2)+'\n')
    print(json.dumps({k:v for k,v in event.items() if k!='stages'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','checkpoint','dataset','oracle-runtime','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();recover(a.root.resolve(),a.checkpoint.resolve(),a.dataset.resolve(),a.oracle_runtime.resolve(),a.out.resolve())
