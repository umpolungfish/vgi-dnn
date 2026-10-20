#!/usr/bin/env python3
"""Suppress only local graph updates in an otherwise identical CPU batch."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from export_joint_retention import build
from collect_joint_retention import run as collect
from measure_composed_retention import measure
from verify_joint_retention import verify


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='VGI/cpu_local_update_control_20261017', type=Path)
    parser.add_argument('--learning-rate', default=.001, type=float)
    parser.add_argument('--keep-local-updates', action='store_true')
    args = parser.parse_args()
    if not 0 < args.learning_rate <= .001:
        raise ValueError('learning rate must be positive and no larger than the baseline')
    root = Path(__file__).resolve().parents[3]
    baseline = root / 'VGI/corpus_cuda_recovered_banked_campaign_20261017/batch_000000'
    out = root / args.out
    out.mkdir(exist_ok=False)
    source = out / 'input'
    source.mkdir()
    rows = [json.loads(line) for line in (baseline / 'input/records.jsonl').open()]
    changed = []
    for index, row in enumerate(rows):
        if row['kind'] == 4 and not args.keep_local_updates:
            changed.append({'index': index, 'previous_steps': row['steps']})
            row['steps'] = 0
    records = source / 'records.jsonl'
    records.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    (source / 'manifest.json').write_bytes((baseline / 'input/manifest.json').read_bytes())
    parent = root / 'VGI/recovered_banked_graph_memory_r2_20261017/returned_checkpoint'
    binding = {'status': 'prepared', 'baseline_records_sha256': sha(baseline / 'input/records.jsonl'),
               'control_records_sha256': sha(records), 'parent_model_sha256': sha(parent / 'model.safetensors'),
               'changed_records': changed, 'learning_rate': args.learning_rate,
               'scope': 'same record order, payloads and anchor schedule; requested steps and learning rate recorded explicitly'}
    (out / 'binding.json').write_text(json.dumps(binding, indent=2) + '\n')
    joint = out / 'joint'
    build(root, parent, records, root / 'VGI/joint_retention_training_graphs_20261016/records.jsonl',
          root / 'VGI/graph_flow_oracle_build_20261016/bare/x86_64-unknown-linux-gnu/release/libig_graph_flow_oracle.a',
          root / 'VGI/graph_flow_dataset_r2_20261016/vocabulary.json', joint, 4., 8, 128, True, True, 64, args.learning_rate)
    verify(root, joint)
    assert (joint / 'graph_anchors.json').read_bytes() == (baseline / 'joint/graph_anchors.json').read_bytes()
    runtime = joint / 'runtime'
    config = json.loads((root / 'VGI/corpus_cuda_recovered_banked_campaign_20261017/campaign.json').read_text())
    with (out / 'profile_launcher.log').open('w') as log:
        subprocess.run([str(root / 'Vox/target/release/vox'), 'profile-native', str(runtime / 'model'),
                        str(runtime / 'corpus_profile'), '--registers', '--duration-ms',
                        str(config['configuration']['duration_ms'])], stdout=log, stderr=subprocess.STDOUT, check=True)
    collect(joint)
    measure(root, [baseline / 'joint/runtime/returned_checkpoint', runtime / 'returned_checkpoint'],
            baseline / 'trajectories', root / 'VGI/graph_flow_dataset_r2_20261016',
            root / 'VGI/training_views_artifact_ready_r4_20261016/validation.jsonl', out / 'retention.json')
    binding['status'] = 'complete'
    (out / 'binding.json').write_text(json.dumps(binding, indent=2) + '\n')


if __name__ == '__main__':
    main()
