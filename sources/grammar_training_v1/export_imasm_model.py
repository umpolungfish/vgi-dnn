#!/usr/bin/env python3
"""Bind trained DT weights to a static executable and lift through Vox."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import torch
from safetensors.torch import load_file


def run(command, timeout=60):
    result = subprocess.run([str(p) for p in command], capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'{command[0]} returned {result.returncode}: {result.stderr}')
    return result.stdout + result.stderr


def block_fields():
    fields = [('norm1', 'norm1.gamma'), ('norm2', 'norm2.gamma'),
              ('gate1', 'gate1'), ('gate2', 'gate2'), ('lam', 'attn.lam')]
    for short, prefix in [('q', 'attn.q'), ('k', 'attn.k'), ('v', 'attn.v'), ('o', 'attn.o'),
                           ('up', 'ffn.up'), ('down', 'ffn.down')]:
        fields.extend((short + suffix, prefix + '.' + suffix) for suffix in ('A', 'B', 'b'))
    return fields


def export(root, checkpoint, output):
    output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((checkpoint / 'config.json').read_text())
    state = load_file(str(checkpoint / 'model.safetensors'), device='cpu')
    probe = json.loads((checkpoint / 'execution_input.json').read_text())
    tokens = probe['tokens']
    lines = [f'#define {name} {value}' for name, value in {
        'V': cfg['vocab_size'], 'D': cfg['d_model'], 'H': cfg['n_heads'],
        'FF': cfg['d_ff'], 'LAYERS': cfg['n_layers'], 'MAXLEN': cfg['max_len'],
        'TOKENS': len(tokens), 'BLEND': int(cfg['residual'] == 'blend'),
        'CAUSAL': int(cfg['causal'])}.items()]
    lines.append('static int tokens[]={' + ','.join(map(str, tokens)) + '};')

    arrays = {}

    def array(name, key):
        arrays[key] = (name, 'F64', state[key].numel()*8)
        values = state[key].double().flatten().tolist()
        storage = 'static double ' if name in ('headA', 'headB', 'headb') else 'static const double '
        lines.append(storage + name + '[]={' + ','.join(format(v, '.17g') for v in values) + '};')

    for name, key in [('embedding_t', 'E_t'), ('embedding_f', 'E_f'), ('position', 'pe'),
                       ('final_norm', 'final_norm.gamma'), ('headA', 'head.A'),
                       ('headB', 'head.B'), ('headb', 'head.b')]:
        array(name, key)
    fields = block_fields()
    for layer in range(cfg['n_layers']):
        expected_modes = torch.tensor([h % 2 == 0 for h in range(cfg['n_heads'])])
        assert torch.equal(state[f'blocks.{layer}.attn.is_join'].flatten(), expected_modes)
        name = f'b{layer}_modes'
        key = f'blocks.{layer}.attn.is_join'
        lines.append('static const unsigned char ' + name + '[]={' +
                     ','.join(str(int(v)) for v in state[key].flatten()) + '};')
        arrays[key] = (name, 'BOOL', state[key].numel())
        for name, key in fields:
            array(f'b{layer}_{name}', f'blocks.{layer}.{key}')
    lines.append('typedef struct {' + ''.join('const double *' + name + ';' for name, _ in fields) + '} BlockWeights;')
    lines.append('static const BlockWeights blocks[]={' + ','.join(
        '{' + ','.join(f'b{layer}_{name}' for name, _ in fields) + '}'
        for layer in range(cfg['n_layers'])) + '};')
    if set(arrays) != set(state):
        raise RuntimeError('checkpoint exporter does not cover all tensors')
    offset, header = 0, {}
    for key in sorted(arrays):
        name, dtype, length = arrays[key]
        header[key] = {'dtype': dtype, 'shape': list(state[key].shape),
                       'data_offsets': [offset, offset+length]}
        offset += length
    header['__metadata__'] = {'architecture': 'dialetheic_transformer',
        'reentry': 'shared_final_trained_block', 'update': 'resident_paired_head_sgd',
        'parent_sha256': hashlib.sha256((checkpoint / 'model.safetensors').read_bytes()).hexdigest()}
    encoded = json.dumps(header, ensure_ascii=True, separators=(',', ':')).encode('utf8')
    encoded += b' ' * (-len(encoded) % 8)
    lines.append('static const unsigned long checkpoint_header_length=' + str(len(encoded)) + ';')
    lines.append('static const unsigned char checkpoint_header[]={' + ','.join(map(str, encoded)) + '};')
    lines.append('static long checkpoint_write(const void*,unsigned long);')
    lines.append('static long emit_checkpoint(void){long total=0,n;')
    for name, length in [('&checkpoint_header_length', 8), ('checkpoint_header', len(encoded))] + [
            (arrays[key][0], arrays[key][2]) for key in sorted(arrays)]:
        lines.append(f'n=checkpoint_write({name},{length});if(n<0)return -1;total+=n;')
    lines.append('return total;}')
    (output / 'weights.h').write_text('\n'.join(lines) + '\n')
    template = Path(__file__).with_name('dialetheic_runtime.c')
    source = output / 'model.c'
    source.write_bytes(template.read_bytes())
    flags = ['-O1', '-fno-builtin', '-fno-stack-protector', '-fno-tree-vectorize',
             '-ffp-contract=off', '-mno-avx', '-mno-avx2', '-mno-fma']
    executable, library = output / 'model', output / 'model.so'
    commands = [ ['cc', *flags, '-nostdlib', '-static', '-no-pie', '-Wl,-e,vgi_eval', source, '-o', executable],
                 ['cc', *flags, '-shared', '-fPIC', source, '-o', library] ]
    for command in commands:
        run(command)
    sys.path.insert(0, str(root / 'bob3ect/reentry_dnn'))
    from dtransformer_torch import DialetheicTransformerTorch, export_to_python
    model = DialetheicTransformerTorch(**cfg).double()
    model.load_state_dict(state)
    py = export_to_python(model)
    lib = ctypes.CDLL(str(library.resolve()))
    lib.vgi_value.argtypes = [ctypes.c_long, ctypes.c_long]
    lib.vgi_value.restype = ctypes.c_double
    lib.vgi_eval.argtypes = [ctypes.c_long]
    lib.vgi_eval.restype = ctypes.c_long
    lib.vgi_settlement.restype = ctypes.c_long
    parity = {}
    components = 2 * len(tokens) * cfg['vocab_size']
    for mode in (0, 1):
        expected = py.forward(tokens, settle=bool(mode))
        flat = [v for matrix in (expected.support, expected.refute) for row in matrix for v in row]
        error = max(abs(lib.vgi_value(i, mode) - value) for i, value in enumerate(flat))
        if error > 1e-6:
            raise RuntimeError(f'full native forward parity failed: mode={mode} max_error={error}')
        parity[str(mode)] = {'max_abs_error': error, 'components_checked': components,
                            'expected_digest': lib.vgi_eval(mode),
                            'settlement': expected.settlement.kind if mode else None}
    module = output / 'model.imasm'
    vox = root / 'Vox/target/release/vox'
    run([vox, 'imasm', executable, '--output', module])
    vm_records = []
    for mode in (0, 1):
        command = [vox, 'run', 'vgi_eval', module, '--args', str(mode)]
        observed = run(command)
        (output / f'vox_mode_{mode}.log').write_text(observed)
        matched = re.search(r'(?:=>|=)\s*(-?\d+)', observed)
        if not matched:
            raise RuntimeError('VM did not return a scalar digest: ' + observed)
        actual = int(matched[1])
        if actual != parity[str(mode)]['expected_digest']:
            raise RuntimeError(f'VM digest mismatch in mode {mode}: {actual}')
        vm_records.append({'command': list(map(str, command)), 'returned_digest': actual,
                           'mode': 'settled_reentry' if mode else 'forward', 'output': observed})
    bindings = {}
    for path in (checkpoint / 'model.safetensors', checkpoint / 'config.json',
                 checkpoint / 'execution_input.json', template, source,
                 output / 'weights.h', executable, module, vox):
        with path.open('rb') as handle:
            bindings[str(path)] = hashlib.file_digest(handle, 'sha256').hexdigest()
    report = {'status': 'passed', 'native_parity': parity, 'vox_execution': vm_records,
              'bindings': bindings, 'compiler_commands': [list(map(str, c)) for c in commands],
              'construction': 'trained parameters and complete forward/re-entry compiled into static image; lifted and executed by Vox',
              'input_scope': 'one provenance-bound held-out token sequence',
              'remaining': 'broader input battery, learned distinction policy, IMASM parameter-update execution'}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'native_parity': parity, 'vox_execution': vm_records}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    export(args.root.resolve(), args.checkpoint.resolve(), args.out.resolve())
