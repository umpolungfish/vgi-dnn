#!/usr/bin/env python3
"""Return composed GPU graph updates to the source-indexed corpus campaign."""
import argparse
import ctypes
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from composed_graph_reference import forward
from export_composed_graph import build
from optimizer_state import split_state
from retained_conditioning import evaluate
from typed_tuple_learning import evaluate_record


def source_evaluate(model, source, typed_tuples=False):
    if typed_tuples and source.get('task') in ('catalog_to_tuple', 'ob3ect_to_tuple'):
        return evaluate_record(model, source)
    return evaluate(model, source['prefix'], source['target'], context=16, checkpointed=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, indent=2) + '\n')
    pending.replace(path)


def adam_step(model, state, parameters, learning_rate, clip):
    """Use the returned bias powers and parameter ordering of the native learner."""
    named = dict(model.named_parameters())
    gradients = torch.cat([named[p['key']].grad.flatten() for p in parameters])
    if not torch.isfinite(gradients).all():
        raise ValueError('nonfinite composed gradient')
    norm = gradients.norm()
    scale = min(1., clip / float(norm)) if clip > 0 and norm > 0 else 1.
    gradients = gradients * scale
    state['clock'] += 1
    state['powers'] *= torch.tensor([.9, .999], dtype=torch.double)
    device = gradients.device
    mean, variance = state['m'].to(device), state['v'].to(device)
    mean += .1 * (gradients - mean)
    variance = .999 * variance + .001 * gradients.square()
    delta = learning_rate * (mean / (1 - state['powers'][0].item())) / (
        (variance / (1 - state['powers'][1].item())).sqrt() + 1e-8)
    position = 0
    with torch.no_grad():
        for entry in parameters:
            parameter = named[entry['key']]
            parameter -= delta[position:position + entry['size']].reshape_as(parameter)
            position += entry['size']
    state['m'], state['v'] = mean.cpu(), variance.cpu()
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('nonfinite composed update')
    return float(norm)


def carrier(result, row, device, clock):
    events = []
    for event in result['trace']:
        probabilities = event['probability'].detach().cpu().tolist()
        proposal = sum(int(value > .5) << bit for bit, value in enumerate(probabilities))
        events.append({k: event[k] for k in ('pass', 'edge', 'input', 'target')} | {
            'proposal': proposal, 'missing': event['target'] & ~proposal & 15,
            'additional': proposal & ~event['target'] & 15,
            'probabilities': probabilities,
            'memory': [v.detach().cpu().flatten().tolist() for v in event['memory']]})
    return {'format': 'vgi_composed_graph_return_v1', 'graph': row, 'device': device,
            'optimizer_clock': clock, 'returned': result['returned'],
            'loss': float(result['loss'].detach()), 'observations': events,
            'memory_t': [m[0].detach().cpu().flatten().tolist() for m in result['memory']],
            'memory_f': [m[1].detach().cpu().flatten().tolist() for m in result['memory']],
            'probabilities': [p.detach().cpu().tolist() for p in result['probabilities']]}


def anchor_loss(model, bank, index, count, weight, oracle):
    """Keep complementary source evidence in the same gradient as the graph."""
    observations = []
    objective = None
    for i in range(count):
        anchor = bank['anchors'][(index * count + i) % len(bank['anchors'])]
        q = anchor['query']
        bits = oracle(q[2], q[3] - 266, q[4] - 266, q[6], q[5])
        assert bits == anchor['source_bits'] and anchor['split'] == 'train'
        result = evaluate(model, q, [], context=16, kind=2, bits=bits)
        term = weight * sum(result['loss_windows']) / (count * result['labels'])
        objective = term if objective is None else objective + term
        support, refutation = result['first_logits']
        proposal = sum(int((refutation if bit % 2 else support)[259 + bit // 2] > 0) << bit for bit in range(4))
        observations.append({'graph_id': anchor['graph_id'], 'edge': anchor['edge'], 'query': q,
            'source_bits': bits, 'proposal_bits': proposal, 'missing': bits & ~proposal & 15,
            'additional': proposal & ~bits & 15, 'loss': result['loss']})
    return objective, observations


def train(root, checkpoint, dataset, library, out, count=32, steps=1, seed=83,
          devices=('cuda:0', 'cuda:1'), passes=2, local_weight=.25,
          anchor_weight=0., anchor_count=8, source_records=None, source_weight=0., typed_tuples=False):
    if count < 1 or steps < 0 or passes < 1 or local_weight < 0 or anchor_weight < 0 or anchor_count < 1 or source_weight < 0:
        raise ValueError('invalid composed training bounds')
    torch.set_num_threads(1)
    sys.path.insert(0, str(root / 'bob3ect/reentry_dnn'))
    from dtransformer_torch import DialetheicTransformerTorch
    if any(not device.startswith('cuda:') for device in devices):
        raise ValueError('production composed backend requires CUDA devices')
    # Compile and lift the same carrier and native source oracle consumed by
    # corpus execution. CUDA performs the differentiable graph transformations.
    build(root, checkpoint, library, dataset / 'vocabulary.json', out,
          passes=passes, steps=steps, local_weight=local_weight)
    metadata = json.loads((out / 'build.json').read_text())
    vocabulary = json.loads((dataset / 'vocabulary.json').read_text())
    rows = [json.loads(line) for line in (dataset / 'transitions.jsonl').open()]
    rows = [row for row in rows if row['split'] == 'train']
    random.Random(seed).shuffle(rows)
    if count > len(rows):
        raise ValueError('trajectory count exceeds training partition')
    selected = rows[:count]
    (out / 'training_graphs.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in selected))
    weights, optimizer = split_state(load_file(str(checkpoint / 'model.safetensors')))
    optimizer = {key: value.clone() for key, value in optimizer.items()}
    initial_clock = int(optimizer['clock'][0])
    source_pool = []
    if source_weight:
        if source_records is None:
            raise ValueError('complete source rehearsal requires source records')
        source_pool = [row for row in map(json.loads, source_records.read_text().splitlines()) if row['kind'] == 0]
        if not source_pool or any(row.get('training_split', 'train') != 'train' for row in source_pool):
            raise ValueError('complete training source records required')
        (out / 'source_rehearsal_records.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in source_pool))
    bank = json.loads((checkpoint / 'joint_gradient_bank.json').read_text()) if anchor_weight else None
    if bank:
        if anchor_count % 2:
            raise ValueError('balanced oracle rehearsal needs an even anchor count')
        for index in range(count):
            masks = [bank['anchors'][(index * anchor_count + i) % len(bank['anchors'])]['source_bits'] for i in range(anchor_count)]
            assert all(sum(bool(mask & (1 << bit)) for mask in masks) == anchor_count // 2 for bit in range(4))
    binding = ctypes.CDLL(str(out / 'model.so'))
    oracle = binding.ig_flow_gate
    oracle.argtypes = [ctypes.c_uint] * 5
    oracle.restype = ctypes.c_uint
    models = {device: DialetheicTransformerTorch(**metadata['config']).double().to(device)
              for device in devices}
    returned = out / 'returned_checkpoint'
    shutil.copytree(checkpoint, returned)
    receipts = []
    anchor_observations = []
    source_observations = []
    for index, row in enumerate(selected):
        device = devices[index % len(devices)]
        model = models[device]
        model.load_state_dict(weights)
        source = source_pool[index % len(source_pool)] if source_pool else None
        for _ in range(steps):
            model.zero_grad(set_to_none=True)
            result = forward(model, row, vocabulary, oracle, passes, local_weight)
            objective = result['loss']
            if source is not None:
                source_result = source_evaluate(model, source, typed_tuples)
                objective = objective + source_weight * sum(source_result['loss_windows']) / source_result['labels']
            if bank:
                loss, observations = anchor_loss(model, bank, index, anchor_count, anchor_weight, oracle)
                objective = objective + loss
                anchor_observations.append({'trajectory': index, 'observations': observations})
            objective.backward()
            adam_step(model, optimizer, metadata['parameters'], metadata['learning_rate'],
                      metadata['stream_gradient_clip'])
        with torch.no_grad():
            result = forward(model, row, vocabulary, oracle, passes, local_weight)
            if source is not None:
                source_after = source_evaluate(model, source, typed_tuples)
        torch.cuda.synchronize(device)
        weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        state = carrier(result, row, device, int(optimizer['clock'][0]))
        if source is not None:
            observation = {'trajectory': index, 'record': source,
                'loss_after': source_after['loss'], 'source_tokens_processed': source_after['source_tokens_processed'],
                'labels': source_after['labels'], 'optimizer_clock': int(optimizer['clock'][0]),
                'memory_t': source_after['memory'][0].flatten().cpu().tolist(),
                'memory_f': source_after['memory'][1].flatten().cpu().tolist()}
            if 'typed_observations' in source_after:
                observation['typed_observations'] = source_after['typed_observations']
                observation['objective'] = source_after['objective']
            state['source_rehearsal'] = observation
            source_observations.append(observation)
            write(returned / 'composed_source_state.json', observation)
        tensors = weights | {'__optimizer.' + key: value for key, value in optimizer.items()}
        save_file(tensors, str(returned / 'model.safetensors.pending'))
        (returned / 'model.safetensors.pending').replace(returned / 'model.safetensors')
        write(out / f'trajectory_{index:04d}.json', state)
        write(returned / 'composed_graph_state.json', state)
        receipts.append({'id': row['id'], 'family': row['graph']['family'], 'split': 'train',
                         'device': device, 'updates': steps, 'loss': state['loss'],
                         'return_bits': state['returned'], 'source_return_bits': row['return_bits'],
                         'state_sha256': sha(out / f'trajectory_{index:04d}.json')})
        print(json.dumps({'graph': index, 'device': device, 'loss': state['loss']}), flush=True)
    lineage_path = returned / 'dataset_lineage.json'
    lineage = json.loads(lineage_path.read_text())
    lineage.setdefault('composed_graph_updates', []).append({
        'parent_checkpoint': str(checkpoint), 'parent_checkpoint_sha256': sha(checkpoint / 'model.safetensors'),
        'source': str(dataset / 'transitions.jsonl'), 'source_sha256': sha(dataset / 'transitions.jsonl'),
        'training_ids': [row['id'] for row in selected], 'steps': steps, 'passes': passes,
        'local_weight': local_weight, 'devices': list(devices), 'updates': count * steps,
        'carried_state_sha256': sha(returned / 'composed_graph_state.json')})
    if bank:
        write(returned / 'composed_anchor_observations.json', anchor_observations)
        lineage['composed_graph_updates'][-1].update(anchor_weight=anchor_weight, anchor_count=anchor_count,
            anchor_bank_sha256=sha(checkpoint / 'joint_gradient_bank.json'),
            anchor_observations_sha256=sha(returned / 'composed_anchor_observations.json'))
    if source_pool:
        write(returned / 'composed_source_observations.json', source_observations)
        lineage['composed_graph_updates'][-1].update(source_weight=source_weight,
            source_records=str(source_records), source_records_sha256=sha(source_records),
            source_observations_sha256=sha(returned / 'composed_source_observations.json'),
            typed_tuples=typed_tuples)
    write(lineage_path, lineage)
    assert int(optimizer['clock'][0]) == initial_clock + count * steps
    assert (returned / 'corpus_cursor.json').read_bytes() == (checkpoint / 'corpus_cursor.json').read_bytes()
    report = {'status': 'passed', 'execution': 'CUDA composed graph learning with lifted native source oracle',
              'backend': 'cuda-composed', 'training_graphs': receipts, 'total_updates': count * steps,
              'optimizer_clock_exact': True, 'parent_checkpoint_sha256': sha(checkpoint / 'model.safetensors'),
              'checkpoint_sha256': sha(returned / 'model.safetensors'),
              'module_sha256': sha(out / 'model.imasm'), 'source_sha256': sha(dataset / 'transitions.jsonl'),
              'devices': {device: torch.cuda.get_device_name(device) for device in devices},
              'passes': passes, 'local_weight': local_weight,
              'anchor_weight': anchor_weight, 'anchor_count': anchor_count,
              'source_weight': source_weight,
              'typed_tuples': typed_tuples,
              'source_records': str(source_records) if source_records else None,
              'source_records_sha256': sha(source_records) if source_records else None,
              'implementation': {name: sha(Path(__file__).with_name(name)) for name in
                  ('train_cuda_trajectories.py','composed_graph_reference.py','retained_conditioning.py',
                   'typed_tuple_learning.py','bootstrap.py')},
              'transformer_sha256': sha(root / 'bob3ect/reentry_dnn/dtransformer_torch.py'),
              'torch_version': torch.__version__,
              'scope': 'complete graph gradients through continuous edge memory; train partition only; corpus cursor preserved'}
    metadata['cuda_composed'] = {key: report[key] for key in ('devices', 'passes', 'local_weight', 'anchor_weight', 'anchor_count')}
    write(out / 'build.json', metadata)
    write(out / 'trajectory_training_receipt.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'checkpoint', 'dataset', 'library', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--count', type=int, default=32)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--seed', type=int, default=83)
    parser.add_argument('--devices', nargs='+', default=['cuda:0', 'cuda:1'])
    parser.add_argument('--passes', type=int, default=2)
    parser.add_argument('--local-weight', type=float, default=.25)
    parser.add_argument('--anchor-weight', type=float, default=0.)
    parser.add_argument('--anchor-count', type=int, default=8)
    parser.add_argument('--source-records', type=Path)
    parser.add_argument('--source-weight', type=float, default=0.)
    parser.add_argument('--typed-tuples', action='store_true')
    args = parser.parse_args()
    train(*(getattr(args, name).resolve() for name in ('root', 'checkpoint', 'dataset', 'library', 'out')),
          args.count, args.steps, args.seed, args.devices, args.passes, args.local_weight,
          args.anchor_weight, args.anchor_count,
          args.source_records.resolve() if args.source_records else None, args.source_weight, args.typed_tuples)
