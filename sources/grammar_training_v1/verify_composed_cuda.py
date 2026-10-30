#!/usr/bin/env python3
"""Execute retained graph learning on each CUDA device against the CPU reference."""
import argparse
import ctypes
import hashlib
import json
import sys
from pathlib import Path
import torch
from safetensors.torch import save_file
from composed_graph_reference import forward
from optimizer_state import load_model
from torch_reference_optimizer import make_optimizer


def run(root, runtime, dataset, out):
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    sys.path.insert(0, str(root/'bob3ect/reentry_dnn'))
    from dtransformer_torch import DialetheicTransformerTorch
    build = json.loads((runtime/'build.json').read_text())
    checkpoint = Path(build['parent'])/'model.safetensors'
    vocabulary = json.loads((dataset/'vocabulary.json').read_text())
    rows = [json.loads(line) for line in (dataset/'transitions.jsonl').open()]
    selected = [next(r for r in rows if r['split']=='train' and r['graph']['family']==family and r['seed_bits']==seed)
                for family,seed in [('chain:AREV',1),('nested:2:1:AREV',3),('feedback:port_order=0',3)]]
    library = ctypes.CDLL(str(runtime/'model.so'))
    oracle = library.ig_flow_gate
    oracle.argtypes = [ctypes.c_uint]*5
    oracle.restype = ctypes.c_uint
    results = []
    assert torch.cuda.is_available(), 'CUDA access required'
    for row in selected:
        baseline = None
        for device in ['cpu'] + [f'cuda:{i}' for i in range(torch.cuda.device_count())]:
            model = DialetheicTransformerTorch(**build['config']).double().to(device)
            model.load_state_dict(load_model(checkpoint))
            optimizer = make_optimizer(model, build)
            for state in optimizer.state.values():
                for key in ('exp_avg', 'exp_avg_sq'):
                    state[key] = state[key].to(device)
            optimizer.zero_grad()
            result = forward(model, row, vocabulary, oracle, 2, 0.)
            result['loss'].backward()
            gradients = {k:p.grad.detach().cpu().clone() for k,p in model.named_parameters()}
            torch.nn.utils.clip_grad_norm_(model.parameters(), build['stream_gradient_clip'])
            optimizer.step()
            state = {k:p.detach().cpu().clone() for k,p in model.named_parameters()}
            loss = float(result['loss'].detach())
            if baseline is None:
                baseline = (loss, gradients, state)
            error = max(float((gradients[k]-baseline[1][k]).abs().max()) for k in gradients)
            parameter_error = max(float((state[k]-baseline[2][k]).abs().max()) for k in state)
            assert abs(loss-baseline[0])<1e-8 and error<1e-8 and parameter_error<1e-8
            tensors = dict(state)
            for key in ('exp_avg', 'exp_avg_sq'):
                tensors['__cuda_probe.'+key] = torch.cat([optimizer.state[p][key].detach().cpu().flatten() for p in model.parameters()])
            path = out/(row['graph']['family'].replace(':','_')+'_'+device.replace(':','_')+'.safetensors')
            save_file(tensors, str(path))
            results.append({'graph_id':row['id'],'family':row['graph']['family'],'device':device, 'loss':loss,
                            'gradient_error':error,'parameter_error':parameter_error,
                            'returned_checkpoint':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    report = {'status':'passed','source_checkpoint':str(checkpoint),
              'source_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'graphs':[r['id'] for r in selected],
              'scope':'one terminal-loss graph-wide Adam update for chain, nested, and feedback training graphs from retained weights and moments on CPU and each GPU',
              'results':results}
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','runtime','dataset','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    run(*(getattr(args,n).resolve() for n in ('root','runtime','dataset','out')))
