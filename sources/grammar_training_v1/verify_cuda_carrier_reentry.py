#!/usr/bin/env python3
"""Compare fresh-process carried graph continuation with an in-memory replay."""
import argparse
import ctypes
import json
import subprocess
import sys
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
from composed_graph_reference import forward
from optimizer_state import split_state
from train_cuda_trajectories import adam_step, carrier, sha


def continuation(root, runtime, dataset, device, live=False):
    torch.set_num_threads(1)
    sys.path.insert(0, str(root / 'bob3ect/reentry_dnn'))
    from dtransformer_torch import DialetheicTransformerTorch
    build = json.loads((runtime / 'build.json').read_text())
    report = json.loads((runtime / 'trajectory_training_receipt.json').read_text())
    checkpoint = runtime / 'returned_checkpoint'
    saved = json.loads((checkpoint / 'composed_graph_state.json').read_text())
    model = DialetheicTransformerTorch(**build['config']).double().to(device)
    weights, optimizer = split_state(load_file(str(checkpoint / 'model.safetensors')))
    model.load_state_dict(weights)
    optimizer = {key: value.clone() for key, value in optimizer.items()}
    library = ctypes.CDLL(str(runtime / 'model.so'))
    oracle = library.ig_flow_gate; oracle.argtypes = [ctypes.c_uint] * 5; oracle.restype = ctypes.c_uint
    vocabulary = json.loads((dataset / 'vocabulary.json').read_text())
    initial = saved
    if live:
        with torch.no_grad():
            result = forward(model, saved['graph'], vocabulary, oracle, report['passes'], report['local_weight'])
        initial = carrier(result, saved['graph'], device, int(optimizer['clock'][0]))
    model.zero_grad(set_to_none=True)
    result = forward(model, saved['graph'], vocabulary, oracle, report['passes'], report['local_weight'], initial=initial)
    before = float(result['loss'].detach())
    result['loss'].backward()
    adam_step(model, optimizer, build['parameters'], build['learning_rate'], build['stream_gradient_clip'])
    with torch.no_grad():
        after = forward(model, saved['graph'], vocabulary, oracle, report['passes'], report['local_weight'], initial=initial)
        reset = forward(model, saved['graph'], vocabulary, oracle, report['passes'], report['local_weight'])
    tensors = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    tensors.update({'__optimizer.' + key: value for key, value in optimizer.items()})
    return tensors, carrier(after, saved['graph'], device, int(optimizer['clock'][0])), before, float(reset['loss'])


def main(args):
    root, runtime, dataset, out = (getattr(args, name).resolve() for name in ('root', 'runtime', 'dataset', 'out'))
    if args.worker:
        tensors, state, before, reset = continuation(root, runtime, dataset, args.device)
        save_file(tensors, str(out / 'fresh_process.safetensors'))
        (out / 'fresh_process.json').write_text(json.dumps({'state': state, 'before': before, 'reset_loss': reset}, indent=2) + '\n')
        return
    out.mkdir(exist_ok=False)
    expected, state, before, reset = continuation(root, runtime, dataset, 'cpu', live=True)
    subprocess.run([sys.executable, str(Path(__file__).resolve()), '--root', str(root), '--runtime', str(runtime),
        '--dataset', str(dataset), '--out', str(out), '--device', args.device, '--worker'], check=True)
    actual = load_file(str(out / 'fresh_process.safetensors'))
    result = json.loads((out / 'fresh_process.json').read_text())
    error = max(float((value.double() - actual[key].double()).abs().max()) for key, value in expected.items())
    memory_error = max(abs(a-b) for key in ('memory_t','memory_f','probabilities')
        for x,y in zip(state[key], result['state'][key]) for a,b in zip(x,y))
    assert max(error, memory_error, abs(before - result['before'])) < 1e-8
    reset_difference = abs(state['loss'] - reset)
    assert reset_difference > 1e-10, 'reset-memory control did not discriminate this graph'
    report = {'status': 'passed', 'fresh_process_device': args.device, 'tensor_error': error,
        'memory_error': memory_error, 'reset_memory_loss_difference': reset_difference,
        'checkpoint_sha256': sha(runtime / 'returned_checkpoint/model.safetensors'),
        'scope': 'fresh process consumes saved continuous graph memory, probabilities, source graph, model and Adam carrier; compared with live CPU continuation and reset control'}
    (out / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root','runtime','dataset','out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--worker', action='store_true')
    main(parser.parse_args())
