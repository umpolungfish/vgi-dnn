#!/usr/bin/env python3
"""Collect joint-gradient source returns with their retained graph-loss provenance."""
import argparse
import hashlib
import json
from pathlib import Path
from collect_retained_batch import collect


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(experiment, interrupted=False):
    runtime = experiment / 'runtime'; build = json.loads((runtime / 'build.json').read_text())
    gradient = json.loads((runtime / 'joint_gradient_verification.json').read_text())
    assert gradient['status'] == 'passed' and gradient['module_sha256'] == sha(runtime / 'model.imasm')
    joint = build['joint_retention']; anchors_path = Path(joint['anchor_records'])
    assert sha(anchors_path) == joint['anchor_records_sha256']
    assert sha(Path(joint['anchor_source'])) == joint['anchor_source_sha256']
    anchors = json.loads(anchors_path.read_text())
    sources = {row['id']: row for row in map(json.loads, Path(joint['anchor_source']).read_text().splitlines())}
    for anchor in anchors:
        source = sources[anchor['graph_id']]
        assert source['split'] == anchor['split'] == 'train'
        assert source['edge_values'][anchor['edge']] == anchor['source_bits']
    collect(runtime, experiment / 'input', runtime / 'corpus_profile', interrupted=interrupted)
    returned = runtime / 'returned_checkpoint'; receipt = json.loads((runtime / 'corpus_training_receipt.json').read_text())
    rows = [json.loads(line) for line in (experiment / 'input/records.jsonl').open()]
    committed=receipt['records']
    language_updates = sum(row['kind'] == 0 for row in rows[:committed])
    observations = [];uncommitted_observations=[]
    if joint.get('execution_trace'):
        expected = [(record, (record * joint['anchors'] + i) % len(anchors))
                    for record,row in enumerate(rows[:committed]) if row['kind'] == 0 for i in range(joint['anchors'])]
        for line in (runtime / 'corpus_profile.stderr').read_text().splitlines():
            if not line.startswith('vgi-joint '):continue
            record,query,target,prediction,loss = map(int, line.split()[1:])
            assert target == anchors[query]['source_bits'] and 0 <= prediction <= 15 and loss >= 0
            observation={'record': record, 'query': query, 'source_bits': target,
                                 'proposal_bits': prediction, 'loss_millionths': loss,
                                 'missing': target & ~prediction & 15, 'additional': prediction & ~target & 15}
            (observations if record < committed else uncommitted_observations).append(observation)
        assert [(row['record'],row['query']) for row in observations] == expected
    bank = {'format': 'vgi_joint_gradient_bank_v1', 'anchors': anchors,
            'graph_weight': joint['weight'], 'source_binding': joint,
            'source_records': str(experiment / 'input/records.jsonl'),
            'source_records_sha256': sha(experiment / 'input/records.jsonl'),
            'source_updates': language_updates, 'source_records_count': receipt['records'],
            'execution_observations': observations,
            'uncommitted_execution_observations': uncommitted_observations,
            'profile_interrupted': interrupted and receipt['status'] != 'complete',
            'gradient_verification': gradient,
            'module_sha256': build['module_sha256']}
    (returned / 'joint_gradient_bank.json').write_text(json.dumps(bank, indent=2) + '\n')
    lineage_path = returned / 'dataset_lineage.json'; lineage = json.loads(lineage_path.read_text())
    lineage.setdefault('joint_gradient_updates', []).append(
        {'bank_sha256': sha(returned / 'joint_gradient_bank.json'),
         'bank_path': str(returned / 'joint_gradient_bank.json'), 'graph_weight': joint['weight'],
         'source_records_sha256': bank['source_records_sha256'], 'source_updates': language_updates,
         'graph_queries': [row['query'] for row in anchors],
         'graph_ids': [row['graph_id'] for row in anchors],
         'parent_checkpoint_sha256': receipt['parent_checkpoint_sha256']})
    lineage_path.write_text(json.dumps(lineage, indent=2) + '\n')
    receipt['dataset_lineage_sha256'] = sha(lineage_path)
    receipt['joint_gradient_bank_sha256'] = sha(returned / 'joint_gradient_bank.json')
    (runtime / 'corpus_training_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'status': receipt['status'], 'records': receipt['records'], 'weight': joint['weight'],
                      'graph_bank_retained': True, 'checkpoint': str(returned)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--interrupted',action='store_true',help='recover the last complete saved capsule when the profiler status is absent')
    a = p.parse_args(); run(a.experiment.resolve(),a.interrupted)
