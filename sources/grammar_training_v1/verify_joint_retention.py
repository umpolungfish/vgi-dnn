#!/usr/bin/env python3
"""Compare a joint source/graph Adam step with independent full-network gradients."""
import argparse
import ctypes
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
import torch

from optimizer_state import load_model
from retained_conditioning import evaluate
from torch_reference_optimizer import make_optimizer


def verify(root, experiment, vm=False):
    torch.set_num_threads(1)
    sys.path.insert(0, str(root / 'bob3ect/reentry_dnn'))
    from dtransformer_torch import DialetheicTransformerTorch
    runtime = experiment / 'runtime'; build = json.loads((runtime / 'build.json').read_text())
    joint = build['joint_retention']; anchors_path = Path(joint['anchor_records'])
    assert hashlib.sha256(anchors_path.read_bytes()).hexdigest() == joint['anchor_records_sha256']
    pool = json.loads(anchors_path.read_text()); assert all(row['split'] == 'train' for row in pool)
    rows = [json.loads(line) for line in (experiment / 'input/records.jsonl').open()]
    index = min((i for i,row in enumerate(rows) if row['kind'] == 0), key=lambda i: len(rows[i]['prefix']) + len(rows[i]['target']))
    row = rows[index]
    anchors = [pool[(index * joint['anchors'] + i) % len(pool)] for i in range(joint['anchors'])]
    model = DialetheicTransformerTorch(**build['config']).double()
    model.load_state_dict(load_model(Path(build['parent']) / 'model.safetensors'))
    optimizer = make_optimizer(model, build); optimizer.zero_grad()
    primary = evaluate(model, row['prefix'], row['target'], context=16)
    objective = sum(primary['loss_windows']) / primary['labels']
    lib = ctypes.CDLL(str(runtime / 'model.so')); lib.vgi_retained_reset()
    for name, n in [('vgi_joint_corpus_learn', 1), ('vgi_corpus_learn', 3)]:
        getattr(lib, name).argtypes = [ctypes.c_long] * n; getattr(lib, name).restype = ctypes.c_long
    for name, n in [('vgi_full_parameter', 1), ('vgi_full_gradient', 1), ('vgi_optimizer_observe', 2), ('vgi_retained_memory', 2)]:
        getattr(lib, name).argtypes = [ctypes.c_long] * n; getattr(lib, name).restype = ctypes.c_double
    lib.ig_flow_gate.argtypes = [ctypes.c_uint] * 5; lib.ig_flow_gate.restype = ctypes.c_uint
    for anchor in anchors:
        q = anchor['query']
        exact = lib.ig_flow_gate(q[2], q[3] - 266, q[4] - 266, q[6], q[5])
        assert exact == anchor['source_bits']
        result = evaluate(model, q, [], context=16, kind=2, bits=exact)
        objective = objective + joint['weight'] * sum(result['loss_windows']) / (result['labels'] * len(anchors))
    objective.backward()
    expected_gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
    norm = float(torch.sqrt(sum((p.grad * p.grad).sum() for p in model.parameters())))
    scale = min(1., build['stream_gradient_clip'] / norm) if norm else 1.
    for p in model.parameters():
        p.grad.mul_(scale)
    optimizer.step()
    assert lib.vgi_joint_corpus_learn(-1) == -120
    assert lib.vgi_optimizer_observe(0, 0) == build['optimizer_initial_clock']
    observed_loss = lib.vgi_joint_corpus_learn(index)
    assert abs(observed_loss / 1e6 - primary['loss']) < 1e-6
    expected = dict(model.named_parameters()); position = 0; gradient_error = parameter_error = moment_error = 0.
    actual_parameters = []; actual_moments = {1: [], 2: []}
    for entry in build['parameters']:
        key = entry['key']; size = entry['size']; p = expected[key]
        actual = torch.tensor([lib.vgi_full_parameter(i) for i in range(position, position + size)], dtype=torch.double)
        gradient = torch.tensor([lib.vgi_full_gradient(i) for i in range(position, position + size)], dtype=torch.double)
        gradient_error = max(gradient_error, float((gradient - expected_gradients[key].flatten()).abs().max()))
        parameter_error = max(parameter_error, float((actual - p.detach().flatten()).abs().max()))
        actual_parameters.extend(actual.tolist())
        for kind, name in ((1, 'exp_avg'), (2, 'exp_avg_sq')):
            actual = torch.tensor([lib.vgi_optimizer_observe(kind, i) for i in range(position, position + size)], dtype=torch.double)
            moment_error = max(moment_error, float((actual - optimizer.state[p][name].flatten()).abs().max()))
            actual_moments[kind].extend(actual.tolist())
        position += size
    assert max(gradient_error, parameter_error, moment_error) < 1e-8, (gradient_error, parameter_error, moment_error)
    assert lib.vgi_optimizer_observe(0, 0) == build['optimizer_initial_clock'] + 1
    with torch.no_grad():
        after = evaluate(model, row['prefix'], row['target'], context=16)
    memory_error = max(abs(lib.vgi_retained_memory(channel, i) - after['memory'][channel].flatten()[i].item())
                       for channel in (0, 1) for i in range(model.d))
    assert memory_error < 1e-8
    control_exact = None
    if joint['weight'] == 0:
        lib.vgi_retained_reset(); assert lib.vgi_corpus_learn(index, 1, 0) >= 0
        assert actual_parameters == [lib.vgi_full_parameter(i) for i in range(position)]
        for kind in (1, 2):
            assert actual_moments[kind] == [lib.vgi_optimizer_observe(kind, i) for i in range(position)]
        control_exact = True
    report = {'status': 'passed', 'source_record': row['example_id'], 'source_record_index': index,
              'source_tokens': len(row['prefix']) + len(row['target']), 'graph_anchors': len(anchors),
              'graph_anchor_pool': len(pool),
              'graph_weight': joint['weight'], 'gradient_max_error': gradient_error,
              'parameter_max_error': parameter_error, 'moment_max_error': moment_error,
              'returned_memory_max_error': memory_error, 'one_adam_step': True,
              'zero_weight_matches_original_update_exactly': control_exact,
              'module_sha256': hashlib.sha256((runtime / 'model.imasm').read_bytes()).hexdigest(),
              'scope': 'same-parameter complete source and graph gradients; validation records excluded from updates'}
    if vm:
        from export_full_learning import compile_model
        probe = runtime / 'joint_process_vm_probe'; probe.mkdir(exist_ok=False)
        cpp = probe / 'full_model.cpp'
        body = (runtime / 'full_model.cpp').read_text()
        body = body[:body.rindex('extern "C" __attribute__((force_align_arg_pointer)) void _start()')]
        cpp.write_text(body + '''
extern "C" __attribute__((force_align_arg_pointer)) void _start(){
    vgi_retained_reset();long result=vgi_joint_corpus_learn(RECORD_INDEX);long status=result<0?2:0;
    long digest=parameter_digest();if(checkpoint_write(&digest,8)<0)status=3;
    __asm__ volatile("syscall"::"a"(60L),"D"(status):"rcx","r11","memory");__builtin_unreachable();}
'''.replace('RECORD_INDEX', str(index)))
        commands = compile_model(root, cpp, probe, 'O2', Path(build['graph_library']))
        native = subprocess.run([str(probe / 'model')], capture_output=True, check=True)
        assert len(native.stdout) == 8
        (probe / 'native.stdout').write_bytes(native.stdout)
        expected_digest = int.from_bytes(native.stdout, 'little', signed=True)
        command = [str(root / 'Vox/target/release/vox'), 'run', str(probe / 'model.imasm')]
        environment = dict(os.environ, VOX_STEPS='2000000000', VOX_PROGRESS='1')
        with (probe / 'execution.log').open('w') as log, (probe / 'vm.stdout').open('wb') as output:
            try:
                result = subprocess.run(command, stdout=output, stderr=log, env=environment, timeout=180)
                text = (probe / 'execution.log').read_text()
                if result.returncode == 0:
                    assert (probe / 'vm.stdout').read_bytes() == native.stdout
                    report['vm'] = {'status': 'passed', 'parameter_digest': expected_digest, 'command': command}
                elif 'step budget' in text:
                    report['vm'] = {'status': 'instruction_bound', 'instruction_limit': 2000000000, 'command': command}
                else:
                    raise AssertionError(text[-2000:])
            except subprocess.TimeoutExpired:
                report['vm'] = {'status': 'observation_time_bound', 'seconds': 180,
                                'instruction_limit': 2000000000, 'command': command}
        (probe / 'build.json').write_text(json.dumps({'commands': commands,
            'source_sha256': hashlib.sha256(cpp.read_bytes()).hexdigest(),
            'module_sha256': hashlib.sha256((probe / 'model.imasm').read_bytes()).hexdigest(),
            'parent_runtime': str(runtime)}, indent=2) + '\n')
    (runtime / 'joint_gradient_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--vm', action='store_true', help='also compare the complete joint-update parameter digest through Vox')
    a = p.parse_args(); verify(a.root.resolve(), a.experiment.resolve(), a.vm)
