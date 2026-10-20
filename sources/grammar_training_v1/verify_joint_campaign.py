#!/usr/bin/env python3
"""Verify corpus-to-graph-to-corpus checkpoint and provenance continuity."""
import argparse
import hashlib
import json
from pathlib import Path
from safetensors.torch import load_file
from optimizer_state import split_state


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(campaign):
    state = json.loads((campaign / 'campaign.json').read_text())
    if not state['configuration'].get('trajectories'):
        raise ValueError('campaign has no graph trajectory integration')
    if state.get('inflight'):
        raise ValueError('verify a returned campaign boundary')
    previous = None; observations = []
    for batch in state['completed_batches']:
        corpus_receipt_path = Path(batch['receipt'])
        runtime = corpus_receipt_path.parent
        corpus = json.loads(corpus_receipt_path.read_text())
        build = json.loads((runtime / 'build.json').read_text())
        parent = Path(build['parent'])
        if previous is not None:
            assert parent == previous
        assert sha(parent / 'model.safetensors') == corpus['parent_checkpoint_sha256']
        _, initial_optimizer = split_state(load_file(str(parent / 'model.safetensors')))
        assert build['optimizer_initial_clock'] == int(initial_optimizer['clock'][0])
        corpus_return = runtime / 'returned_checkpoint'
        assert sha(corpus_return / 'model.safetensors') == corpus['returned_checkpoint_sha256']
        for state_file in ('composed_graph_state.json','composed_source_state.json',
                           'composed_source_observations.json','composed_anchor_observations.json',
                           'composed_graph_bank.json'):
            if (parent / state_file).exists():
                assert (parent / state_file).read_bytes() == (corpus_return / state_file).read_bytes()
        if state['configuration'].get('joint_gradient'):
            parent_lineage=json.loads((parent / 'dataset_lineage.json').read_text())
            corpus_lineage=json.loads((corpus_return / 'dataset_lineage.json').read_text())
            assert corpus_lineage['joint_gradient_updates'][:-1] == parent_lineage.get('joint_gradient_updates',[])
            assert {k:v for k,v in parent_lineage.items() if k!='joint_gradient_updates'} == {k:v for k,v in corpus_lineage.items() if k!='joint_gradient_updates'}
            bank=corpus_return / 'joint_gradient_bank.json'
            assert sha(bank) == corpus_lineage['joint_gradient_updates'][-1]['bank_sha256']
            joint_bank=json.loads(bank.read_text())
            assert len(joint_bank['execution_observations']) == joint_bank['source_updates'] * joint_bank['source_binding']['anchors']
        else:
            assert (parent / 'dataset_lineage.json').read_bytes() == (corpus_return / 'dataset_lineage.json').read_bytes()
        graph_receipt_path = Path(batch['trajectory_receipt'])
        graph = json.loads(graph_receipt_path.read_text())
        graph_runtime = graph_receipt_path.parent; returned = graph_runtime / 'returned_checkpoint'
        graph_build = json.loads((graph_runtime / 'build.json').read_text())
        assert Path(graph_build['parent']) == corpus_return
        assert graph['parent_checkpoint_sha256'] == corpus['returned_checkpoint_sha256']
        assert sha(returned / 'model.safetensors') == graph['checkpoint_sha256'] == batch['checkpoint_sha256']
        assert graph['total_updates'] == batch['trajectory_updates']
        assert (corpus_return / 'corpus_cursor.json').read_bytes() == (returned / 'corpus_cursor.json').read_bytes()
        for name in ('config.json', 'tokenizer.json', 'execution_input.json'):
            assert (corpus_return / name).read_bytes() == (returned / name).read_bytes()
        if (corpus_return / 'joint_gradient_bank.json').exists():
            assert (corpus_return / 'joint_gradient_bank.json').read_bytes() == (returned / 'joint_gradient_bank.json').read_bytes()
        before = json.loads((corpus_return / 'dataset_lineage.json').read_text())
        after = json.loads((returned / 'dataset_lineage.json').read_text())
        history_key = 'composed_graph_updates' if graph.get('backend') == 'cuda-composed' else 'graph_trajectory_updates'
        history = before.get(history_key, [])
        assert after[history_key][:-1] == history
        assert {k:v for k,v in after.items() if k != history_key} == {k:v for k,v in before.items() if k != history_key}
        latest = after[history_key][-1]
        assert latest['parent_checkpoint_sha256'] == graph['parent_checkpoint_sha256']
        selected = [json.loads(line) for line in (graph_runtime / 'training_graphs.jsonl').open()]
        assert all(row['split'] == 'train' for row in selected)
        assert latest['training_ids'] == [row['id'] for row in selected] == [row['id'] for row in graph['training_graphs']]
        assert latest['updates'] == graph['total_updates']
        if graph.get('backend') == 'cuda-composed':
            from verify_cuda_trajectory_return import verify as verify_cuda
            dataset = Path(state['configuration']['trajectories']['bindings']['transitions.jsonl']['path']).parent
            verify_cuda(Path(__file__).resolve().parents[3], graph_runtime, dataset)
        _, before_optimizer = split_state(load_file(str(corpus_return / 'model.safetensors')))
        _, after_optimizer = split_state(load_file(str(returned / 'model.safetensors')))
        assert int(after_optimizer['clock'][0]) == int(before_optimizer['clock'][0]) + graph['total_updates']
        observations.append({'batch': batch['batch'], 'corpus_records': corpus['records'],
                             'graph_trajectories': len(selected), 'graph_updates': graph['total_updates'],
                             'corpus_cursor_exact_through_graph_return': True,
                             'prior_exposure_and_new_graph_ids_retained': True,
                             'optimizer_clock_exact': True, 'checkpoint_sha256': graph['checkpoint_sha256']})
        previous = returned
    assert previous is not None and Path(state['checkpoint']) == previous
    assert Path(state['cursor']) == previous / 'corpus_cursor.json'
    result = {'status': 'passed', 'campaign': str(campaign), 'batches': observations,
              'checkpoint': str(previous), 'scope': 'actual returned corpus and graph checkpoints, Adam clock, cursor bytes, and exposure lineage'}
    (campaign / 'joint_return_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    a = p.parse_args(); verify(a.campaign.resolve())
