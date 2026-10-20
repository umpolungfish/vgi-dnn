#!/usr/bin/env python3
"""Train source-bound graph trajectories through Vox CPU execution and return."""
import argparse
import ctypes
import hashlib
import json
import random
import shutil
import struct
import subprocess
from pathlib import Path

from export_graph_learning import build
from export_full_learning import compile_model
from graph_format import decode
from retained_format import capsule_end


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train(root, checkpoint, dataset, library, out, count, steps, seed, duration_ms, collect_only=False, mode=2):
    if count < 1 or not 0 <= steps <= 100:
        raise ValueError('positive trajectory count and zero to 100 correction steps required')
    rows = [json.loads(line) for line in (dataset / 'transitions.jsonl').open()]
    candidates = [row for row in rows if row['split'] == 'train']
    random.Random(seed).shuffle(candidates)
    if count > len(candidates):
        raise ValueError('trajectory count exceeds training partition')
    selected = candidates[:count]
    if collect_only:
        report = json.loads((out / 'build.json').read_text())
        assert report['parent'] == str(checkpoint)
        assert report['graph_curriculum'] == {
            'split': 'train', 'count': count, 'steps': steps, 'seed': seed, 'mode': mode,
            'source_sha256': sha(dataset / 'transitions.jsonl'),
            'selected_sha256': sha(out / 'training_graphs.jsonl')}
        assert selected == [json.loads(line) for line in (out / 'training_graphs.jsonl').open()]
        assert report['source_sha256'] == sha(out / 'full_model.cpp')
        assert report['module_sha256'] == sha(out / 'model.imasm')
        return collect(checkpoint, dataset, out, selected, steps, mode)
    build(root, checkpoint, out, library, dataset / 'vocabulary.json', mode, steps, 0, 'adam')
    (out / 'training_graphs.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in selected))
    ids = {row['name']: row['id'] for row in json.loads((dataset / 'vocabulary.json').read_text())['tokens']}
    source = out / 'full_model.cpp'
    body = source.read_text()
    body = body[:body.rindex('extern "C" __attribute__((force_align_arg_pointer)) void _start()')]
    calls = []
    for row in selected:
        graph = row['graph']; incoming = {}; outgoing = {}
        calls.append(f'if(vgi_graph_reset({len(graph["nodes"])},{len(graph["edges"])},{graph["readout"]})<0)return -1;')
        for i, name in enumerate(graph['nodes']):
            calls.append(f'if(vgi_graph_node({i},{ids[name]})<0)return -2;')
        for i, (a, b) in enumerate(graph['edges']):
            calls.append(f'if(vgi_graph_edge({i},{a},{b},{outgoing.get(a,0)},{incoming.get(b,0)})<0)return -3;')
            outgoing[a] = outgoing.get(a, 0) + 1; incoming[b] = incoming.get(b, 0) + 1
        # The source oracle computes every edge target on the actual current
        # graph input. No expected output is supplied to the learned proposal.
        calls.append(f'if(vgi_graph_run({row["seed_bits"]},{mode},0,{steps},0)<0)return -4;')
        calls.append('if(vgi_graph_save()<0)return -5;')
    body += '\nstatic long graph_curriculum(){\n' + '\n'.join(calls) + '\nreturn 0;}\n'
    body += '''extern "C" __attribute__((force_align_arg_pointer)) void _start(){
    vgi_retained_reset();long result=graph_curriculum();long status=result<0?2:0;
    __asm__ volatile("syscall"::"a"(60L),"D"(status):"rcx","r11","memory");__builtin_unreachable();}
'''
    source.write_text(body)
    report = json.loads((out / 'build.json').read_text())
    report['trajectory_compile_commands'] = compile_model(root, source, out, 'O2', library)
    report.update(source_sha256=sha(source), module_sha256=sha(out / 'model.imasm'),
                  graph_curriculum={'split': 'train', 'count': count, 'steps': steps, 'seed': seed, 'mode': mode,
                                    'source_sha256': sha(dataset / 'transitions.jsonl'),
                                    'selected_sha256': sha(out / 'training_graphs.jsonl')})
    (out / 'build.json').write_text(json.dumps(report, indent=2) + '\n')
    profile = out / 'trajectory_profile'
    command = [str(root / 'Vox/target/release/vox'), 'profile-native', str(out / 'model'),
               str(profile), '--registers', '--duration-ms', str(duration_ms)]
    with (out / 'profile_launcher.log').open('wb') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    collect(checkpoint, dataset, out, selected, steps, mode)


def collect(checkpoint, dataset, out, selected, steps, mode):
    count = len(selected)
    ids = {row['name']: row['id'] for row in json.loads((dataset / 'vocabulary.json').read_text())['tokens']}
    profile = out / 'trajectory_profile'
    status = dict(line.split('=', 1) for line in profile.with_suffix('.status').read_text().splitlines())
    data = profile.with_suffix('.stdout').read_bytes(); receipts = []; completed = 0; total_updates = 0
    oracle = ctypes.CDLL(str(out / 'model.so')).ig_flow_gate
    oracle.argtypes = [ctypes.c_uint] * 5; oracle.restype = ctypes.c_uint
    returned = out / 'returned_checkpoint'; returned.mkdir()
    while data:
        retained_end = capsule_end(data)
        n, e = struct.unpack_from('<2q', data, retained_end + 8)
        end = retained_end + 176 + 16 * n + 64 * e
        capsule = data[:end]
        weights, retained, graph = decode(capsule); data = data[end:]
        row = selected[completed]; expected = row['graph']
        assert graph['opcodes'] == [ids[name] for name in expected['nodes']]
        assert list(zip(graph['from'], graph['to'])) == [tuple(edge) for edge in expected['edges']]
        assert graph['seed'] == row['seed_bits'] and graph['mode'] == mode
        source_exact = graph['node_inputs'] == row['node_inputs'] and graph['values'] == row['edge_values']
        if mode == 2:
            assert source_exact and graph['diagnostics'][11] == row['return_bits'] and graph['diagnostics'][0] == 1
        # Validate the oracle at the learner's actual inputs, including states
        # outside the source-guided trajectory in mode 3.
        # The compiled runtime exports the same linked ig_flow_gate used during production.
        for edge, node in enumerate(graph['from']):
            fan = graph['from'].count(node)
            assert oracle(graph['opcodes'][node], graph['node_inputs'][node], graph['seed'], graph['outport'][edge], fan) == graph['expected'][edge]
        (returned / 'model.safetensors').write_bytes(weights)
        (returned / 'retained_state.json').write_text(json.dumps(retained, indent=2) + '\n')
        (returned / 'graph_state.json').write_text(json.dumps(graph, indent=2) + '\n')
        # Keep the complete return of every trajectory for re-entry and review.
        (out / f'trajectory_{completed:04d}.stdout').write_bytes(capsule)
        updates = graph['diagnostics'][5]; total_updates += updates
        receipts.append({'id': row['id'], 'family': expected['family'], 'split': row['split'],
                         'seed': row['seed_bits'], 'passes': graph['diagnostics'][1],
                         'updates': updates, 'before_edge_errors': graph['diagnostics'][3],
                         'after_edge_errors': graph['diagnostics'][4], 'return_bits': graph['diagnostics'][11],
                         'source_return_bits': row['return_bits'], 'stable': bool(graph['diagnostics'][0]),
                         'source_flow_exact': source_exact, 'local_oracle_exact': True, 'return_bytes': end})
        completed += 1
    assert completed == count and status['wait_status'] == '0' and status['deadline_reached'] == 'false'
    from safetensors.torch import load_file
    import torch
    from optimizer_state import split_state
    initial, old_optimizer = split_state(load_file(str(checkpoint / 'model.safetensors')))
    actual, optimizer = split_state(load_file(str(returned / 'model.safetensors')))
    assert int(optimizer['clock'][0]) == int(old_optimizer['clock'][0]) + total_updates
    assert all(bool(t.isfinite().all()) for t in (*actual.values(), *optimizer.values()))
    changed = any(not torch.equal(initial[key], actual[key]) for key in initial)
    assert changed == (total_updates > 0)
    if total_updates == 0:
        assert all(torch.equal(old_optimizer[key], optimizer[key]) for key in optimizer)
    for name in ('config.json', 'tokenizer.json', 'execution_input.json', 'corpus_cursor.json', 'dataset_lineage.json', 'joint_gradient_bank.json', 'composed_graph_state.json', 'composed_anchor_observations.json', 'composed_source_state.json', 'composed_source_observations.json', 'composed_graph_bank.json'):
        if (checkpoint / name).exists():
            shutil.copyfile(checkpoint / name, returned / name)
    lineage = json.loads((returned / 'dataset_lineage.json').read_text()) if (returned / 'dataset_lineage.json').exists() else {'format': 'vgi_dataset_lineage_v1', 'transitions': []}
    lineage.setdefault('graph_trajectory_updates', []).append(
        {'parent_checkpoint': str(checkpoint), 'parent_checkpoint_sha256': sha(checkpoint / 'model.safetensors'),
         'source': str(dataset / 'transitions.jsonl'), 'source_sha256': sha(dataset / 'transitions.jsonl'),
         'training_ids': [r['id'] for r in selected], 'steps': steps, 'mode': mode, 'updates': total_updates})
    (returned / 'dataset_lineage.json').write_text(json.dumps(lineage, indent=2) + '\n')
    result = {'status': 'passed', 'execution': 'Vox profile-native CPU', 'profile_status': status,
              'training_graphs': receipts, 'total_updates': total_updates,
              'parameters_changed': changed, 'optimizer_clock_exact': True,
              'checkpoint_sha256': sha(returned / 'model.safetensors'),
              'parent_checkpoint_sha256': sha(checkpoint / 'model.safetensors'),
              'module_sha256': sha(out / 'model.imasm'), 'source_sha256': sha(dataset / 'transitions.jsonl'),
              'mode': mode, 'scope': 'training graph trajectories with local source checks and retained learned proposals'}
    (out / 'trajectory_training_receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': 'passed', 'graphs': completed, 'updates': total_updates,
                      'checkpoint': str(returned)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'checkpoint', 'dataset', 'library', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--count', type=int, default=32); p.add_argument('--steps', type=int, default=1)
    p.add_argument('--seed', type=int, default=83); p.add_argument('--duration-ms', type=int, default=300000)
    p.add_argument('--collect-only', action='store_true', help='collect an existing Vox profile return without rerunning it')
    p.add_argument('--mode', type=int, choices=(2, 3), default=2,
                   help='2 propagates source results; 3 propagates the corrected learned proposals')
    a = p.parse_args()
    train(a.root.resolve(), a.checkpoint.resolve(), a.dataset.resolve(), a.library.resolve(),
          a.out.resolve(), a.count, a.steps, a.seed, a.duration_ms, a.collect_only, a.mode)
