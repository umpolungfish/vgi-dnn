#!/usr/bin/env python3
"""Check both language losses, complete gradients, memory and shared Adam return."""
import argparse
import ctypes
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from optimizer_state import split_state
from paired_language_learning import evaluate_record
from retained_conditioning import evaluate
from train_cuda_trajectories import adam_step, sha


def verify(root, runtime):
    torch.set_num_threads(1)
    sys.path.insert(0, str(root / 'bob3ect/reentry_dnn'))
    from dtransformer_torch import DialetheicTransformerTorch
    build = json.loads((runtime / 'build.json').read_text())
    weights, optimizer = split_state(load_file(str(Path(build['parent']) / 'model.safetensors')))
    model = DialetheicTransformerTorch(**build['config']).double()
    model.load_state_dict(weights)
    row = build['paired_language']['record']
    result = evaluate_record(model, row)
    objective = sum(result['loss_windows']) / result['labels']
    objective.backward()
    named = dict(model.named_parameters())
    gradient = torch.cat([named[p['key']].grad.flatten() for p in build['parameters']])
    lib = ctypes.CDLL(str(runtime / 'model.so'))
    for name, count in [('vgi_full_gradient',1),('vgi_full_parameter',1),('vgi_optimizer_observe',2),
                        ('vgi_language_loss',1),('vgi_language_memory',3)]:
        function = getattr(lib, name)
        function.argtypes = [ctypes.c_long] * count
        function.restype = ctypes.c_double
    assert lib.vgi_paired_language_update() == 0
    actual = torch.tensor([lib.vgi_full_gradient(i) for i in range(len(gradient))], dtype=torch.double)
    gradient_error = float((actual-gradient).abs().max())
    memory_error = 0.
    for channel, name in enumerate(('support','refutation')):
        if name not in result['language_branches']:
            continue
        branch = result['language_branches'][name]
        assert abs(lib.vgi_language_loss(channel)-branch['loss']) < 1.1e-6
        for evidence, key in enumerate(('memory_t','memory_f')):
            observed = torch.tensor([lib.vgi_language_memory(channel,evidence,c) for c in range(model.d)],dtype=torch.double)
            memory_error = max(memory_error, float((observed-torch.tensor(branch[key],dtype=torch.double)).abs().max()))
    adam_step(model, optimizer, build['parameters'], build['learning_rate'], build['stream_gradient_clip'])
    expected = torch.cat([named[p['key']].detach().flatten() for p in build['parameters']])
    actual = torch.tensor([lib.vgi_full_parameter(i) for i in range(len(expected))],dtype=torch.double)
    parameter_error = float((actual-expected).abs().max())
    moment_error = 0.
    for kind, key in ((1,'m'),(2,'v'),(3,'powers')):
        expected = optimizer[key].flatten()
        actual = torch.tensor([lib.vgi_optimizer_observe(kind,i) for i in range(len(expected))],dtype=torch.double)
        moment_error = max(moment_error,float((actual-expected).abs().max()))
    assert int(lib.vgi_optimizer_observe(0,0)) == int(optimizer['clock'][0])
    model.zero_grad(set_to_none=True)
    model.load_state_dict(weights)
    omitted = 'refutation' if row['refutation_target'] is not None else 'support'
    retained = [name for name in ('support','refutation') if name != omitted and row[name+'_target'] is not None]
    control = model.E_t.new_zeros(())
    for name in retained:
        from paired_language_learning import Readout
        value = evaluate(Readout(model,0 if name=='support' else 1),row['prefix'],list(row[name+'_target'].encode())+[258],context=16)
        control = control + sum(value['loss_windows'])/value['labels']/len(retained)
    if row.get('evidence_presence'):
        source = evaluate(model,row['prefix'],[],context=16,kind=2,differentiable_state=True)
        support,refutation = source['first_logits']
        logits = torch.stack((support[259],refutation[259]))
        target = logits.new_tensor([row['evidence_bits']&1,(row['evidence_bits']>>1)&1])
        mask = logits.new_tensor(row.get('presence_mask',[1,1]))
        control = control + (F.binary_cross_entropy_with_logits(logits,target,reduction='none')*mask).sum()/mask.sum()
    control.backward()
    positive_gradient = torch.cat([named[p['key']].grad.flatten() for p in build['parameters']])
    omission_delta = float((gradient-positive_gradient).abs().max())
    assert gradient_error < 1e-7 and parameter_error < 1e-8 and moment_error < 1e-8 and memory_error < 1e-8, (gradient_error,parameter_error,moment_error,memory_error)
    assert omission_delta > 1e-8
    report = {'status':'passed','source_record':row.get('id',row['example_id']),'evidence_bits':row['evidence_bits'],
              'gradient_error':gradient_error,'parameter_error':parameter_error,'optimizer_error':moment_error,
              'memory_error':memory_error,'omit_language_branch_gradient_delta':omission_delta,'omitted_branch':omitted,
              'module_sha256':sha(runtime/'model.imasm'),
              'scope':'observed source language continuations and masked evidence; full gradients, paired memories and one shared Adam update'}
    (runtime/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--runtime',type=Path,required=True)
    args=parser.parse_args()
    verify(args.root.resolve(),args.runtime.resolve())
