#!/usr/bin/env python3
"""Compare the executing shared source/graph gradient with CPU and both GPUs."""
import argparse
import ctypes
import json
import subprocess
import sys
from pathlib import Path
import torch
from safetensors.torch import save_file
from composed_graph_reference import forward
from composed_format import decode
from optimizer_state import load_model
from retained_conditioning import evaluate
from torch_reference_optimizer import make_optimizer
from train_cuda_trajectories import sha


def verify(root, experiment, dataset, devices):
    torch.set_num_threads(1)
    sys.path.insert(0, str(root / 'bob3ect/reentry_dnn'))
    from dtransformer_torch import DialetheicTransformerTorch
    runtime = experiment / 'runtime'; build = json.loads((runtime / 'build.json').read_text())
    joint = build['composed_source_joint']; local = build['joint_retention']
    assert sha(experiment / 'input/records.jsonl') == joint['source_records_sha256']
    assert sha(Path(joint['graphs'])) == joint['graphs_sha256']
    rows = [json.loads(line) for line in (experiment / 'input/records.jsonl').open()]
    index = joint['source_index']; row = rows[index]
    graph = json.loads(Path(joint['graphs']).read_text())[joint['graph_index']]
    assert graph['split'] == 'train' and row['kind'] == 0
    bank = json.loads(Path(local['anchor_records']).read_text())
    anchors = [bank[(index * local['anchors'] + i) % len(bank)] for i in range(local['anchors'])]
    vocabulary = json.loads((dataset / 'vocabulary.json').read_text())
    library = ctypes.CDLL(str(runtime / 'model.so')); library.vgi_retained_reset()
    library.ig_flow_gate.argtypes = [ctypes.c_uint] * 5; library.ig_flow_gate.restype = ctypes.c_uint
    library.vgi_joint_corpus_learn.argtypes = [ctypes.c_long]; library.vgi_joint_corpus_learn.restype = ctypes.c_long
    for name,count in [('vgi_full_parameter',1),('vgi_full_gradient',1),('vgi_optimizer_observe',2),('vgi_retained_memory',2),('vgi_composed_observe',4)]:
        getattr(library,name).argtypes = [ctypes.c_long] * count; getattr(library,name).restype = ctypes.c_double
    code = library.vgi_joint_corpus_learn(index)
    assert code >= 0, {'native_joint_status':code}
    n = sum(entry['size'] for entry in build['parameters'])
    native_parameters = torch.tensor([library.vgi_full_parameter(i) for i in range(n)], dtype=torch.double)
    native_gradients = torch.tensor([library.vgi_full_gradient(i) for i in range(n)], dtype=torch.double)
    save_file({'gradient':native_gradients},str(experiment / 'native_joint_gradient.safetensors'))
    native_moments = [torch.tensor([library.vgi_optimizer_observe(kind,i) for i in range(n)], dtype=torch.double) for kind in (1,2)]
    results = []
    for device in devices:
        model = DialetheicTransformerTorch(**build['config']).double().to(device)
        model.load_state_dict(load_model(Path(build['parent']) / 'model.safetensors'))
        optimizer = make_optimizer(model, build)
        for state in optimizer.state.values():
            for key in ('exp_avg','exp_avg_sq'):
                state[key] = state[key].to(device)
        optimizer.zero_grad()
        source = evaluate(model, row['prefix'], row['target'], context=16, checkpointed=True)
        objective = sum(source['loss_windows']) / source['labels']
        for anchor in anchors:
            q = anchor['query']; bits = library.ig_flow_gate(q[2],q[3]-266,q[4]-266,q[6],q[5])
            assert bits == anchor['source_bits']
            result = evaluate(model,q,[],context=16,kind=2,bits=bits)
            objective += local['weight'] * sum(result['loss_windows']) / (len(anchors) * result['labels'])
        composed = forward(model,graph,vocabulary,library.ig_flow_gate,joint['passes'],joint['local_weight'])
        objective += joint['graph_weight'] * composed['loss']
        objective.backward()
        parameters = dict(model.named_parameters())
        gradient = torch.cat([parameters[e['key']].grad.detach().cpu().flatten() for e in build['parameters']])
        norm = float(gradient.norm())
        scale = min(1., build['stream_gradient_clip'] / norm) if norm and build['stream_gradient_clip'] > 0 else 1.
        for parameter in model.parameters():
            parameter.grad.mul_(scale)
        optimizer.step()
        actual = torch.cat([parameters[e['key']].detach().cpu().flatten() for e in build['parameters']])
        moment_error = 0.
        for kind,key in enumerate(('exp_avg','exp_avg_sq')):
            values = torch.cat([optimizer.state[parameters[e['key']]][key].detach().cpu().flatten() for e in build['parameters']])
            moment_error = max(moment_error,float((values-native_moments[kind]).abs().max()))
        with torch.no_grad():
            source_after = evaluate(model,row['prefix'],row['target'],context=16)
            graph_after = forward(model,graph,vocabulary,library.ig_flow_gate,joint['passes'],joint['local_weight'])
        source_memory_error = max(abs(library.vgi_retained_memory(channel,c)-source_after['memory'][channel].flatten()[c].item())
            for channel in (0,1) for c in range(model.d))
        graph_memory_error = max(abs(library.vgi_composed_observe(2,edge,channel,c)-graph_after['memory'][edge][channel].flatten()[c].item())
            for edge in range(len(graph['graph']['edges'])) for channel in (0,1) for c in range(model.d))
        errors = {'gradient':float((gradient-native_gradients).abs().max()),
            'parameters':float((actual-native_parameters).abs().max()),'moments':moment_error,
            'source_memory':source_memory_error,'graph_memory':graph_memory_error}
        assert max(errors.values()) < 1e-8, (device,errors)
        path = experiment / ('joint_' + device.replace(':','_') + '.safetensors')
        tensors = {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
        save_file(tensors,str(path))
        results.append({'device':device,'objective':float(objective.detach()),'source_loss_before':source['loss'],
            'source_loss_after':source_after['loss'],'graph_loss_before':float(composed['loss'].detach()),
            'graph_loss_after':float(graph_after['loss']),'errors':errors,'model_sha256':sha(path)})
    process = subprocess.run([str(runtime / 'model')],capture_output=True,check=True)
    (experiment / 'native_joint_return.stdout').write_bytes(process.stdout)
    (experiment / 'native_joint_return.stderr').write_bytes(process.stderr)
    weights,retained,discrete,continuous = decode(process.stdout)
    assert retained['source_bank'] == row['prefix'] + row['target']
    assert retained['completed_records'] == 1 and retained['committed_updates'] == 1
    assert continuous['source_edges'] == graph['edge_values'] and continuous['source_return'] == graph['return_bits']
    (experiment / 'native_model.safetensors').write_bytes(weights)
    (experiment / 'native_retained_state.json').write_text(json.dumps(retained,indent=2)+'\n')
    (experiment / 'native_composed_state.json').write_text(json.dumps(continuous,indent=2)+'\n')
    assert library.vgi_optimizer_observe(0,0) == build['optimizer_initial_clock']+1
    report = {'status':'passed','source_record':row['example_id'],'graph_id':graph['id'],
        'source_tokens':len(row['prefix'])+len(row['target']),'source_graph_binding_exact':True,
        'one_shared_adam_step':True,'module_sha256':sha(runtime/'model.imasm'),
        'source_checkpoint_sha256':sha(Path(build['parent'])/'model.safetensors'),'results':results,
        'scope':'whole source, balanced source oracle and composed graph losses at identical weights; native gradients and returned source and graph memories compared with CPU and CUDA'}
    (experiment / 'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root','experiment','dataset'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--devices',nargs='+',default=['cpu','cuda:0','cuda:1'])
    a = parser.parse_args(); verify(a.root.resolve(),a.experiment.resolve(),a.dataset.resolve(),a.devices)
