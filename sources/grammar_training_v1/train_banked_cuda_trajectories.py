#!/usr/bin/env python3
"""Train typed/source/graph objectives with source-bound banked re-entry."""
import argparse
import ctypes
import json
import random
import shutil
import sys
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
from composed_graph_reference import forward
from export_composed_graph import build
from optimizer_state import split_state
from train_cuda_trajectories import adam_step,carrier,anchor_loss,sha,write,source_evaluate as original_source_evaluate
import graph_reentry_bank as memory_bank
from cached_typed_tuple_learning import objective as cached_objective


def source_evaluate(model, source, typed_tuples=False):
    if source.get('task') == 'paired_language_evidence':
        from paired_language_learning import evaluate_record
        return evaluate_record(model, source)
    if typed_tuples and source.get('task') in ('catalog_to_tuple','ob3ect_to_tuple'):
        # Use the original framing/binding checks and return protocol, with the
        # shared exact source-prefix objective supplied by this module.
        prefix=source['prefix'];target=source['target']
        if prefix[0]!=256 or prefix[-1]!=257 or target[-1]!=258:
            raise ValueError('invalid typed source framing')
        prompt=bytes(prefix[1:-1]).decode('utf8');address=bytes(target[:-1]).decode('utf8')
        import hashlib
        for key,value in [('prompt_sha256',prompt),('target_sha256',address)]:
            if hashlib.sha256(value.encode()).hexdigest()!=source[key]:
                raise ValueError('typed source binding changed')
        loss,observations=cached_objective(model,prompt,address);last=observations[-1]
        memory=tuple(torch.tensor(last[k],dtype=model.E_t.dtype,device=model.E_t.device).reshape(1,1,-1) for k in ('memory_t','memory_f'))
        return {'loss_windows':[loss*12],'labels':12,'loss':float(loss.detach()),'memory':memory,
                'source_tokens_processed':sum(o['source_tokens_processed'] for o in observations),
                'typed_observations':observations,'objective':'twelve_axis_choices'}
    return original_source_evaluate(model,source,False)

def train(root, checkpoint, dataset, library, out, count=32, steps=1, seed=83,
          devices=('cuda:0', 'cuda:1'), passes=2, local_weight=.25,
          anchor_weight=0., anchor_count=8, source_records=None, source_weight=0., typed_tuples=False, graph_reentry=True,
          distinction_weight=0., distinction_count=8, graph_weight=1.):
    if not graph_reentry:
        raise ValueError('banked trainer requires graph re-entry')
    if not 0 <= graph_weight <= 100:
        raise ValueError('invalid graph objective weight')
    if not 0 <= distinction_weight <= 100 or distinction_count < 1:
        raise ValueError('invalid paired distinction rehearsal bounds')
    if count < 1 or steps < 0 or passes < 1 or local_weight < 0 or anchor_weight < 0 or anchor_count < 1 or source_weight < 0:
        raise ValueError('invalid composed training bounds')
    torch.set_num_threads(1)
    sys.path.insert(0, str(root / 'bob3ect/reentry_dnn'))
    from dtransformer_torch import DialetheicTransformerTorch
    if any(not device.startswith('cuda:') for device in devices):
        raise ValueError('production composed backend requires CUDA devices')
    # Compile and lift the same carrier and native source oracle consumed by
    # corpus execution. CUDA performs the differentiable graph transformations.
    vox_sha256 = sha(root / 'Vox/target/release/vox')
    build(root, checkpoint, library, dataset / 'vocabulary.json', out,
          passes=passes, steps=steps, local_weight=local_weight)
    if sha(root / 'Vox/target/release/vox') != vox_sha256:
        raise ValueError('Vox executable changed during lifting')
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
        source_pool = [row for row in map(json.loads, source_records.read_text().splitlines()) if row['kind'] in (0,6)]
        if not source_pool or any(row.get('training_split', 'train') != 'train' for row in source_pool):
            raise ValueError('complete training source records required')
        (out / 'source_rehearsal_records.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in source_pool))
    graph_bank = memory_bank.load(checkpoint, rows, sha(dataset / 'transitions.jsonl'), sha(dataset / 'vocabulary.json'), metadata['config']['d_model'], initial_clock)
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
    distinction_pool = []
    if distinction_weight:
        from distinction_rehearsal import prepare as prepare_distinctions
        if source_records is None:
            raise ValueError('distinction rehearsal requires source-bound mixed records')
        distinction_pool = prepare_distinctions([json.loads(s) for s in source_records.read_text().splitlines()], binding)
        (out / 'distinction_rehearsal_records.json').write_text(json.dumps(distinction_pool, indent=2) + '\n')
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
        initial_state = memory_bank.initial(graph_bank, row, model.d, int(optimizer['clock'][0]))
        distinction_observations = []
        for _ in range(steps):
            model.zero_grad(set_to_none=True)
            result = forward(model, row, vocabulary, oracle, passes, local_weight, initial=initial_state)
            objective = graph_weight * result['loss']
            if source is not None:
                source_result = source_evaluate(model, source, typed_tuples)
                objective = objective + source_weight * sum(source_result['loss_windows']) / source_result['labels']
            if bank:
                loss, observations = anchor_loss(model, bank, index, anchor_count, anchor_weight, oracle)
                objective = objective + loss
                anchor_observations.append({'trajectory': index, 'observations': observations})
            if distinction_pool:
                from distinction_rehearsal import loss as distinction_loss
                loss, observations = distinction_loss(model, distinction_pool, index, distinction_count, distinction_weight)
                objective = objective + loss
                distinction_observations.append(observations)
            objective.backward()
            adam_step(model, optimizer, metadata['parameters'], metadata['learning_rate'],
                      metadata['stream_gradient_clip'])
        with torch.no_grad():
            result = forward(model, row, vocabulary, oracle, passes, local_weight, initial=initial_state)
            if source is not None:
                source_after = source_evaluate(model, source, typed_tuples)
        torch.cuda.synchronize(device)
        weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        state = carrier(result, row, device, int(optimizer['clock'][0]))
        if distinction_pool:
            state['distinction_rehearsal'] = distinction_observations
        state['reentry_parent_sha256'] = memory_bank.binding(initial_state) if initial_state is not None else None
        if source is not None:
            observation = {'trajectory': index, 'record': source,
                'loss_after': source_after['loss'], 'source_tokens_processed': source_after['source_tokens_processed'],
                'labels': source_after['labels'], 'optimizer_clock': int(optimizer['clock'][0]),
                'memory_t': source_after['memory'][0].flatten().cpu().tolist(),
                'memory_f': source_after['memory'][1].flatten().cpu().tolist()}
            if 'typed_observations' in source_after:
                observation['typed_observations'] = source_after['typed_observations']
                observation['objective'] = source_after['objective']
            if 'language_branches' in source_after:
                observation['language_branches'] = source_after['language_branches']
                observation['objective'] = source_after['objective']
            if 'language_presence' in source_after:
                observation['language_presence'] = source_after['language_presence']
            state['source_rehearsal'] = observation
            source_observations.append(observation)
            write(returned / 'composed_source_state.json', observation)
        tensors = weights | {'__optimizer.' + key: value for key, value in optimizer.items()}
        save_file(tensors, str(returned / 'model.safetensors.pending'))
        (returned / 'model.safetensors.pending').replace(returned / 'model.safetensors')
        write(out / f'trajectory_{index:04d}.json', state)
        write(returned / 'composed_graph_state.json', state)
        memory_bank.store(graph_bank, state, model.d, int(optimizer['clock'][0]))
        write(returned / 'composed_graph_bank.json', graph_bank)
        receipts.append({'id': row['id'], 'family': row['graph']['family'], 'split': 'train',
                         'device': device, 'updates': steps, 'loss': state['loss'],
                         'reentered': initial_state is not None,
                         'return_bits': state['returned'], 'source_return_bits': row['return_bits'],
                         'state_sha256': sha(out / f'trajectory_{index:04d}.json')})
        print(json.dumps({'graph': index, 'device': device, 'loss': state['loss']}), flush=True)
    lineage_path = returned / 'dataset_lineage.json'
    lineage = json.loads(lineage_path.read_text())
    lineage.setdefault('composed_graph_updates', []).append({
        'parent_checkpoint': str(checkpoint), 'parent_checkpoint_sha256': sha(checkpoint / 'model.safetensors'),
        'source': str(dataset / 'transitions.jsonl'), 'source_sha256': sha(dataset / 'transitions.jsonl'),
        'training_ids': [row['id'] for row in selected], 'steps': steps, 'passes': passes,
        'local_weight': local_weight, 'graph_weight': graph_weight, 'devices': list(devices), 'updates': count * steps,
        'carried_state_sha256': sha(returned / 'composed_graph_state.json'),
        'graph_bank_sha256': sha(returned / 'composed_graph_bank.json')})
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
    if distinction_pool:
        lineage['composed_graph_updates'][-1].update(
            distinction_weight=distinction_weight, distinction_count=distinction_count,
            distinction_pool_sha256=sha(out / 'distinction_rehearsal_records.json'),
            distinction_source_records=str(source_records), distinction_source_records_sha256=sha(source_records))
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
              'graph_weight': graph_weight,
              'anchor_weight': anchor_weight, 'anchor_count': anchor_count,
              'source_weight': source_weight,
              'distinction_weight': distinction_weight, 'distinction_count': distinction_count,
              'distinction_pool_sha256': sha(out / 'distinction_rehearsal_records.json') if distinction_pool else None,
              'typed_tuples': typed_tuples,
              'graph_reentry': True, 'cached_typed_prefix': True, 'vox_sha256': vox_sha256,
              'reentered_graphs': sum(r['reentered'] for r in receipts),
              'source_records': str(source_records) if source_records else None,
              'source_records_sha256': sha(source_records) if source_records else None,
              'implementation': {name: sha(Path(__file__).with_name(name)) for name in
                  ('train_banked_cuda_trajectories.py','train_cuda_trajectories.py','graph_reentry_bank.py',
                   'cached_typed_tuple_learning.py','composed_graph_reference.py','retained_conditioning.py',
                   'typed_tuple_learning.py','bootstrap.py','distinction_rehearsal.py','paired_language_learning.py')},
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
    parser.add_argument('--distinction-weight', type=float, default=0.)
    parser.add_argument('--distinction-count', type=int, default=8)
    parser.add_argument('--graph-weight', type=float, default=1.)
    args = parser.parse_args()
    train(*(getattr(args, name).resolve() for name in ('root', 'checkpoint', 'dataset', 'library', 'out')),
          args.count, args.steps, args.seed, args.devices, args.passes, args.local_weight,
          args.anchor_weight, args.anchor_count,
          args.source_records.resolve() if args.source_records else None, args.source_weight, args.typed_tuples,
          distinction_weight=args.distinction_weight, distinction_count=args.distinction_count,
          graph_weight=args.graph_weight)
