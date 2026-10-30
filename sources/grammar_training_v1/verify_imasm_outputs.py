#!/usr/bin/env python3
"""Check individual evidence outputs and re-entry return through the Vox VM."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import re
import subprocess


def verify(root, runtime, checkpoint):
    lib = ctypes.CDLL(str((runtime / 'model.so').resolve()))
    lib.vgi_component.argtypes = [ctypes.c_long, ctypes.c_long]
    lib.vgi_component.restype = ctypes.c_long
    config = json.loads((checkpoint / 'config.json').read_text())
    input_record = json.loads((checkpoint / 'execution_input.json').read_text())
    channel_size = len(input_record['tokens']) * config['vocab_size']
    checks = []
    vox, module = root / 'Vox/target/release/vox', runtime / 'model.imasm'
    for mode, component in [(0, 0), (0, channel_size), (1, channel_size-1), (1, 2*channel_size-1)]:
        expected = lib.vgi_component(component, mode)
        command = [str(vox), 'run', 'vgi_component', str(module), '--args', f'{component},{mode}']
        result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=True)
        output = result.stdout + result.stderr
        match = re.search(r'\)\s*=\s*(-?\d+)', output)
        if not match or int(match[1]) != expected:
            raise RuntimeError('component mismatch: ' + output)
        checks.append({'command': command, 'expected': expected, 'output': output})
    command = [str(vox), 'run', 'vgi_settlement', str(module)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=True)
    output = result.stdout + result.stderr
    match = re.search(r'\)\s*=\s*(-?\d+)', output)
    if not match or int(match[1]) != 1:
        raise RuntimeError('expected stable settlement: ' + output)
    checks.append({'command': command, 'expected': 1, 'output': output})
    report = {'status': 'passed', 'checks': checks,
              'module_sha256': hashlib.sha256(module.read_bytes()).hexdigest(),
              'component_unit': 'logit multiplied by 1000000, truncated toward zero',
              'settlement_codes': {'0': 'divergent', '1': 'stable', '2': 'imaginary'}}
    (runtime / 'component_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    verify(args.root.resolve(), args.runtime.resolve(), args.checkpoint.resolve())
