#!/usr/bin/env python3
"""Produce full-carrier local port targets from the compiled canonical graph gate."""
import argparse,ctypes,hashlib,json,subprocess
from pathlib import Path
from retained_format import stream


def build(root,host,probe,out):
    out.mkdir(parents=True,exist_ok=False)
    vocabulary=json.loads(subprocess.check_output([str(host),'--table'],text=True))
    library=ctypes.CDLL(str(probe/'probe.so'));gate=library.vgi_graph_gate
    gate.argtypes=[ctypes.c_long]*5;gate.restype=ctypes.c_long
    source=root/'MoDoT/imasm_core/src/flow.rs';digest=hashlib.sha256(source.read_bytes()).hexdigest();rows=[]
    for token in vocabulary['tokens']:
        if token['id']>=12:continue
        for before in range(16):
            for seed in range(16):
                for fan in range(1,token['output_ports']+1):
                    for slot in range(fan):
                        args=(token['id'],before,seed,slot,fan);target=gate(*args);assert 0<=target<16
                        identity='graph-gate:'+','.join(map(str,args))
                        rows.append({'prefix':[256,71,token['id'],266+before,266+seed,fan,slot,257],
                            'target':[],'kind':2,'bits':target,'steps':1,'checkpoint_chunks':1,
                            'example_id':identity,'source_id':identity,
                            'family':'graph-gate:'+','.join(map(str,(token['id'],before,slot,fan))),
                            'task':'graph_local_transition','task_ordinal':len(rows),'training_split':'train',
                            'source':{'function':'imasm_core::flow::gate_out','path':str(source.relative_to(root)),'sha256':digest},
                            'operation':token,'before_bits':before,'seed_bits':seed,'slot':slot,'fan':fan,
                            'criterion':'returned full four-atom carrier on this explicit graph port'})
    text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows);payload=stream(rows)
    (out/'records.jsonl').write_text(text);(out/'train.vgir').write_bytes(payload)
    report={'records':len(rows),'tokens':sum(len(r['prefix']) for r in rows),'source':'imasm_core::flow::gate_out',
            'input_and_output_states':16,'all_carrier_shapes_retained':True,'source_sha256':digest,
            'records_sha256':hashlib.sha256(text.encode()).hexdigest(),'input_sha256':hashlib.sha256(payload).hexdigest(),
            'probe_module_sha256':hashlib.sha256((probe/'probe.imasm').read_bytes()).hexdigest(),
            'graph_structure_holdouts':'graph_flow_dataset_r2_20261016; depth four reserved for test'}
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','host','probe','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();build(a.root.resolve(),a.host.resolve(),a.probe.resolve(),a.out.resolve())
