#!/usr/bin/env python3
"""Measure typed value selection separately from constructor type validity."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
from bootstrap import encode_address
from optimizer_state import load_model
from typed_tuple_learning import objective, infer


def measure(root, records, checkpoints, out, limit=8):
    torch.set_num_threads(1);sys.path.insert(0,str(root/'bob3ect/reentry_dnn'))
    from dtransformer_torch import DialetheicTransformerTorch
    selected=[]
    for line in records.open():
        row=json.loads(line)
        if row.get('task')=='catalog_to_tuple':
            encode_address(row['target']);selected.append(row)
            if len(selected)==limit:break
    assert len(selected)==limit and limit>0
    results=[]
    with torch.no_grad():
        for checkpoint in checkpoints:
            model=DialetheicTransformerTorch(**json.loads((checkpoint/'config.json').read_text())).double()
            model.load_state_dict(load_model(checkpoint/'model.safetensors'));model.eval()
            rows=[]
            for source in selected:
                prediction=infer(model,source['prompt'])
                loss,_=objective(model,source['prompt'],source['target'])
                rows.append({'source':source,'prediction':prediction,'loss':float(loss),
                    'correct_coordinates':sum(a==b for a,b in zip(source['target'],prediction['address'])),
                    'exact':source['target']==prediction['address']})
            result={'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256((checkpoint/'model.safetensors').read_bytes()).hexdigest(),
                    'rows':rows,'exact':sum(r['exact'] for r in rows),'coordinates_correct':sum(r['correct_coordinates'] for r in rows),
                    'coordinates':12*len(rows),'mean_axis_loss':sum(r['loss'] for r in rows)/len(rows)}
            results.append(result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    out.write_text(json.dumps({'records':str(records),'records_sha256':hashlib.sha256(records.read_bytes()).hexdigest(),
        'scope':'first bounded catalog tuple sample from supplied validation file; type validity is by construction, value accuracy is measured',
        'results':results},ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','records','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--checkpoints',nargs='+',type=Path,required=True);p.add_argument('--limit',type=int,default=8)
    a=p.parse_args();measure(a.root.resolve(),a.records.resolve(),[c.resolve() for c in a.checkpoints],a.out.resolve(),a.limit)
