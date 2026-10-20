#!/usr/bin/env python3
"""Replay production CUDA graph returns on CPU and check the whole carrier."""
import argparse
import ctypes
import json
import sys
from pathlib import Path
import torch
from safetensors.torch import load_file
from composed_graph_reference import forward
from optimizer_state import split_state
from train_cuda_trajectories import adam_step, anchor_loss, sha, source_evaluate
from retained_conditioning import evaluate
import graph_reentry_bank


def verify(root, runtime, dataset, recorded_devices=False):
    torch.set_num_threads(1)
    sys.path.insert(0, str(root / 'bob3ect/reentry_dnn'))
    from dtransformer_torch import DialetheicTransformerTorch
    build = json.loads((runtime / 'build.json').read_text())
    report = json.loads((runtime / 'trajectory_training_receipt.json').read_text())
    reference_source_evaluate = source_evaluate
    if report.get('cached_typed_prefix'):
        from train_banked_cuda_trajectories import source_evaluate as reference_source_evaluate
    parent = Path(build['parent']); returned = runtime / 'returned_checkpoint'
    assert sha(parent / 'model.safetensors') == report['parent_checkpoint_sha256']
    assert sha(returned / 'model.safetensors') == report['checkpoint_sha256']
    assert sha(dataset / 'transitions.jsonl') == report['source_sha256']
    assert sha(runtime / 'model.imasm') == report['module_sha256']
    selected = [json.loads(line) for line in (runtime / 'training_graphs.jsonl').open()]
    assert len(selected) == len(report['training_graphs']) > 0
    assert sum(row['updates'] for row in report['training_graphs']) == report['total_updates']
    source = {row['id']: row for row in map(json.loads, (dataset / 'transitions.jsonl').read_text().splitlines())}
    assert all(row == source[row['id']] and row['split'] == 'train' for row in selected)
    initial, optimizer = split_state(load_file(str(parent / 'model.safetensors')))
    optimizer = {key: value.clone() for key, value in optimizer.items()}
    model = DialetheicTransformerTorch(**build['config']).double()
    model.load_state_dict(initial)
    lib = ctypes.CDLL(str(runtime / 'model.so'))
    oracle = lib.ig_flow_gate; oracle.argtypes = [ctypes.c_uint] * 5; oracle.restype = ctypes.c_uint
    vocabulary = json.loads((dataset / 'vocabulary.json').read_text())
    state_error = 0.
    bank = json.loads((parent / 'joint_gradient_bank.json').read_text()) if report.get('anchor_weight') else None
    observed_anchors = []
    source_pool = []
    distinction_pool = []
    if report.get('distinction_weight'):
        from distinction_rehearsal import prepare as prepare_distinctions
        source_path = Path(report['source_records'])
        assert sha(source_path) == report['source_records_sha256']
        distinction_pool = prepare_distinctions([json.loads(s) for s in source_path.read_text().splitlines()], lib)
        assert distinction_pool == json.loads((runtime / 'distinction_rehearsal_records.json').read_text())
        assert sha(runtime / 'distinction_rehearsal_records.json') == report['distinction_pool_sha256']
    graph_bank = None
    if report.get('graph_reentry'):
        graph_bank = graph_reentry_bank.load(parent,list(source.values()),sha(dataset/'transitions.jsonl'),
            sha(dataset/'vocabulary.json'),model.d,int(optimizer['clock'][0]))
    if report.get('source_weight'):
        source_path = Path(report['source_records'])
        assert sha(source_path) == report['source_records_sha256']
        source_pool = [row for row in map(json.loads, source_path.read_text().splitlines()) if row['kind'] in (0,6)]
        assert source_pool == [json.loads(line) for line in (runtime / 'source_rehearsal_records.jsonl').open()]
    for i, (row, record) in enumerate(zip(selected, report['training_graphs'])):
        model.to(record['device'] if recorded_devices else 'cpu')
        initial_state = graph_reentry_bank.initial(graph_bank,row,model.d,int(optimizer['clock'][0])) if graph_bank is not None else None
        observed_distinctions = []
        for _ in range(record['updates']):
            model.zero_grad(set_to_none=True)
            result = forward(model, row, vocabulary, oracle, report['passes'], report['local_weight'],initial=initial_state)
            objective = report['graph_weight'] * result['loss'] if 'graph_weight' in report else result['loss']
            if source_pool:
                source = source_pool[i % len(source_pool)]
                source_result = reference_source_evaluate(model,source,report.get('typed_tuples',False))
                objective = objective + report['source_weight'] * sum(source_result['loss_windows']) / source_result['labels']
            if bank:
                loss, observations = anchor_loss(model, bank, i, report['anchor_count'], report['anchor_weight'], oracle)
                objective = objective + loss
                observed_anchors.append({'trajectory': i, 'observations': observations})
            if distinction_pool:
                from distinction_rehearsal import loss as distinction_loss
                loss, observations = distinction_loss(model, distinction_pool, i, report['distinction_count'], report['distinction_weight'])
                objective = objective + loss
                observed_distinctions.append(observations)
            objective.backward()
            adam_step(model, optimizer, build['parameters'], build['learning_rate'], build['stream_gradient_clip'])
        saved = json.loads((runtime / f'trajectory_{i:04d}.json').read_text())
        if distinction_pool:
            actual = saved['distinction_rehearsal']
            assert len(actual) == len(observed_distinctions)
            for actual_step, expected_step in zip(actual, observed_distinctions):
                assert len(actual_step) == len(expected_step) == report['distinction_count']
                for actual_event, expected_event in zip(actual_step, expected_step):
                    assert {k:v for k,v in actual_event.items() if k != 'loss'} == {k:v for k,v in expected_event.items() if k != 'loss'}
                    assert abs(actual_event['loss'] - expected_event['loss']) < 1e-8
        assert saved['graph'] == row and sha(runtime / f'trajectory_{i:04d}.json') == record['state_sha256']
        with torch.no_grad():
            result = forward(model, row, vocabulary, oracle, report['passes'], report['local_weight'],initial=initial_state)
            if source_pool:
                source = source_pool[i % len(source_pool)]
                source_after = reference_source_evaluate(model,source,report.get('typed_tuples',False))
                observed = saved['source_rehearsal']
                assert observed['record'] == source
                assert observed['source_tokens_processed'] == source_after['source_tokens_processed']
                if 'typed_observations' in source_after:
                    assert observed['objective'] == 'twelve_axis_choices'
                    for actual_axis, expected_axis in zip(observed['typed_observations'], source_after['typed_observations']):
                        for key in ('axis','allowed','target','proposal','source_tokens_processed'):
                            assert actual_axis[key] == expected_axis[key]
                        for key in ('scores','memory_t','memory_f'):
                            state_error=max(state_error,float((torch.tensor(actual_axis[key])-torch.tensor(expected_axis[key])).abs().max()))
                if 'language_branches' in source_after:
                    assert observed['objective'] == source_after['objective']
                    assert observed['language_branches'].keys() == source_after['language_branches'].keys()
                    for name, expected_branch in source_after['language_branches'].items():
                        actual_branch = observed['language_branches'][name]
                        for key in ('target','labels','source_tokens_processed'):
                            assert actual_branch[key] == expected_branch[key]
                        assert abs(actual_branch['loss']-expected_branch['loss']) < 1e-8
                        for key in ('memory_t','memory_f'):
                            state_error=max(state_error,float((torch.tensor(actual_branch[key],dtype=torch.double)-torch.tensor(expected_branch[key],dtype=torch.double)).abs().max()))
                if 'language_presence' in source_after:
                    actual_presence = observed['language_presence']
                    expected_presence = source_after['language_presence']
                    for key in ('target_bits','proposal_bits'):
                        assert actual_presence[key] == expected_presence[key]
                    assert actual_presence.get('observed_mask') == expected_presence.get('observed_mask')
                    assert abs(actual_presence['loss']-expected_presence['loss']) < 1e-8, (i,actual_presence,expected_presence)
                    state_error=max(state_error,float((torch.tensor(actual_presence['logits'],dtype=torch.double)-torch.tensor(expected_presence['logits'],dtype=torch.double)).abs().max()))
                assert observed['labels'] == source_after['labels']
                for channel,key in enumerate(('memory_t','memory_f')):
                    state_error = max(state_error,float((source_after['memory'][channel].detach().cpu().flatten()-torch.tensor(observed[key],dtype=torch.double)).abs().max()))
                assert abs(observed['loss_after']-source_after['loss']) < 1e-8
        assert result['returned'] == saved['returned']
        assert len(result['trace']) == len(saved['observations'])
        for event, observation in zip(result['trace'], saved['observations']):
            assert all(event[key] == observation[key] for key in ('pass', 'edge', 'input', 'target'))
            assert observation['missing'] == observation['target'] & ~observation['proposal'] & 15
            assert observation['additional'] == observation['proposal'] & ~observation['target'] & 15
        for edge, pair in enumerate(result['memory']):
            for channel, key in enumerate(('memory_t', 'memory_f')):
                state_error = max(state_error, float((pair[channel].detach().cpu().flatten() - torch.tensor(saved[key][edge], dtype=torch.double)).abs().max()))
            state_error = max(state_error, float((result['probabilities'][edge].detach().cpu() - torch.tensor(saved['probabilities'][edge], dtype=torch.double)).abs().max()))
        if graph_bank is not None:
            assert saved['reentry_parent_sha256'] == (graph_reentry_bank.binding(initial_state) if initial_state is not None else None)
            assert record['reentered'] == (initial_state is not None)
            graph_reentry_bank.store(graph_bank,saved,model.d,int(optimizer['clock'][0]))
    actual, actual_optimizer = split_state(load_file(str(returned / 'model.safetensors')))
    assert all(torch.isfinite(value).all() for value in (*actual.values(), *actual_optimizer.values()))
    assert (actual_optimizer['v'] >= 0).all()
    parameter_error = max(float((value.detach().cpu().double() - actual[key].double()).abs().max()) for key, value in model.state_dict().items())
    moment_error = max(float((value - actual_optimizer[key]).abs().max()) for key, value in optimizer.items())
    assert max(parameter_error, moment_error, state_error) < 1e-8, (parameter_error, moment_error, state_error)
    for file in parent.iterdir():
        overwritten = ('model.safetensors', 'dataset_lineage.json', 'composed_graph_state.json')
        if graph_bank is not None:
            overwritten += ('composed_graph_bank.json',)
        if bank:
            overwritten += ('composed_anchor_observations.json',)
        if source_pool:
            overwritten += ('composed_source_state.json','composed_source_observations.json')
        if file.is_file() and file.name not in overwritten:
            assert file.read_bytes() == (returned / file.name).read_bytes(), file.name
    previous = json.loads((parent / 'dataset_lineage.json').read_text())
    lineage = json.loads((returned / 'dataset_lineage.json').read_text())
    assert lineage['composed_graph_updates'][:-1] == previous.get('composed_graph_updates', [])
    assert {k:v for k,v in lineage.items() if k != 'composed_graph_updates'} == {k:v for k,v in previous.items() if k != 'composed_graph_updates'}
    assert lineage['composed_graph_updates'][-1]['training_ids'] == [row['id'] for row in selected]
    if distinction_pool:
        event = lineage['composed_graph_updates'][-1]
        assert event['distinction_pool_sha256'] == report['distinction_pool_sha256']
        assert event['distinction_source_records_sha256'] == report['source_records_sha256']
        assert event['distinction_weight'] == report['distinction_weight']
        assert event['distinction_count'] == report['distinction_count']
    if bank:
        actual_observations = json.loads((returned / 'composed_anchor_observations.json').read_text())
        assert len(actual_observations) == len(observed_anchors)
        for actual_record, expected_record in zip(actual_observations, observed_anchors):
            assert actual_record['trajectory'] == expected_record['trajectory']
            assert len(actual_record['observations']) == len(expected_record['observations']) == report['anchor_count']
            for actual_event, expected_event in zip(actual_record['observations'], expected_record['observations']):
                assert {k:v for k,v in actual_event.items() if k != 'loss'} == {k:v for k,v in expected_event.items() if k != 'loss'}
                assert abs(actual_event['loss'] - expected_event['loss']) < 1e-8
        assert sha(returned / 'composed_anchor_observations.json') == lineage['composed_graph_updates'][-1]['anchor_observations_sha256']
    if source_pool:
        source_observations = json.loads((returned / 'composed_source_observations.json').read_text())
        assert len(source_observations) == len(selected)
        assert source_observations == [json.loads((runtime / f'trajectory_{i:04d}.json').read_text())['source_rehearsal'] for i in range(len(selected))]
        assert source_observations[-1] == json.loads((returned / 'composed_source_state.json').read_text())
        assert sha(returned / 'composed_source_observations.json') == lineage['composed_graph_updates'][-1]['source_observations_sha256']
    if graph_bank is not None:
        assert json.loads((returned/'composed_graph_bank.json').read_text()) == graph_bank
        assert sha(returned/'composed_graph_bank.json') == lineage['composed_graph_updates'][-1]['graph_bank_sha256']
        assert report['reentered_graphs'] == sum(r['reentered'] for r in report['training_graphs'])
    result = {'status': 'passed', 'graphs': len(selected), 'parameter_error': parameter_error,
              'replay_backend': 'recorded CUDA devices' if recorded_devices else 'CPU',
              'optimizer_error': moment_error, 'carried_state_error': state_error,
              'complete_source_records_rehearsed': len(selected) if source_pool else 0,
              'complete_source_tokens_rehearsed': sum(
                  sum(len(source_pool[i % len(source_pool)]['prefix']) + len(text.encode()) + 1
                      for name in ('support','refutation')
                      if (text := source_pool[i % len(source_pool)].get(name + '_target')) is not None) +
                  (len(source_pool[i % len(source_pool)]['prefix']) if source_pool[i % len(source_pool)].get('evidence_presence') else 0)
                  if source_pool[i % len(source_pool)]['kind'] == 6 else
                  len(source_pool[i % len(source_pool)]['prefix']) + len(source_pool[i % len(source_pool)]['target'])
                  for i in range(len(selected))) if source_pool else 0,
              'checkpoint_sha256': report['checkpoint_sha256'],
              'reentered_graphs': report.get('reentered_graphs',0),
              'scope': ('recorded-device replay' if recorded_devices else 'CPU replay') + ' of shared GPU updates, complete memory and evidence return, exact source banks, corpus cursor and prior lineage'}
    (runtime / 'cuda_return_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'runtime', 'dataset'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--recorded-devices',action='store_true',help='replay on each producing CUDA device; distinct from the independent CPU comparison')
    args = parser.parse_args()
    verify(*(getattr(args, name).resolve() for name in ('root', 'runtime', 'dataset')),recorded_devices=args.recorded_devices)
