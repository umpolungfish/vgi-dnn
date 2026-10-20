#!/usr/bin/env python3
"""Save the complete safetensors checkpoint emitted by an IMASM learning call."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess

import torch
from safetensors.torch import load_file


def save(root, runtime, parent, output, bits=3, steps=32):
    output.mkdir(parents=True, exist_ok=False)
    tokens = json.loads((parent / 'execution_input.json').read_text())['tokens']
    first, second = [sum(t << (9*i) for i,t in enumerate(tokens[start:start+4]))
                     for start in (0,4)]
    command = [str(root/'Vox/target/release/vox'), 'run', 'vgi_learn_export',
               str(runtime/'model.imasm'), '--args', f'{first},{second},{bits},{steps}']
    result = subprocess.run(command, capture_output=True, timeout=60, check=True)
    if len(result.stdout) < 8:
        raise RuntimeError('no checkpoint header returned')
    header_length = struct.unpack('<Q', result.stdout[:8])[0]
    header = json.loads(result.stdout[8:8+header_length])
    payload_length = max(v['data_offsets'][1] for k,v in header.items() if k != '__metadata__')
    length = 8 + header_length + payload_length
    if len(result.stdout) < length:
        raise RuntimeError('truncated checkpoint return')
    checkpoint_path = output / 'model.safetensors'
    checkpoint_path.write_bytes(result.stdout[:length])
    returned = load_file(str(checkpoint_path))
    initial = load_file(str(parent/'model.safetensors'))
    library = ctypes.CDLL(str((runtime/'model.so').resolve()))
    library.vgi_learn.argtypes = [ctypes.c_long]*5
    library.vgi_learn.restype = ctypes.c_long
    digest = library.vgi_learn(first,second,bits,steps,2)
    library.vgi_parameter.argtypes = [ctypes.c_long]
    library.vgi_parameter.restype = ctypes.c_double
    param_index, changed = 0, []
    for key in ('head.A','head.B','head.b'):
        expected = torch.tensor([library.vgi_parameter(i) for i in range(
            param_index,param_index+returned[key].numel())],dtype=torch.double).reshape(returned[key].shape)
        if not torch.equal(returned[key], expected):
            raise RuntimeError('emitted checkpoint differs from executing weights: ' + key)
        param_index += returned[key].numel()
        if not torch.equal(returned[key],initial[key].double()):
            changed.append(key)
    if set(returned)!=set(initial):
        raise RuntimeError('checkpoint lost tensors')
    for key in set(initial)-{'head.A','head.B','head.b'}:
        expected = initial[key] if initial[key].dtype == torch.bool else initial[key].double()
        if not torch.equal(expected,returned[key]):
            raise RuntimeError('retained trunk changed: ' + key)
    if len(changed)!=3:
        raise RuntimeError('expected all paired-head tensors to update')
    for filename in ('config.json','tokenizer.json','execution_input.json'):
        shutil.copyfile(parent/filename,output/filename)
    log = result.stdout[length:].decode('utf8',errors='replace') + result.stderr.decode('utf8',errors='replace')
    (output/'vox_checkpoint.log').write_text(log)
    report = {'status':'passed','command':command,'target_bits':bits,'steps':steps,
              'changed_tensors':changed,'retained_tensors':len(returned)-3,
              'head_parameter_digest':digest,'checkpoint_sha256':hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
              'parent_sha256':hashlib.sha256((parent/'model.safetensors').read_bytes()).hexdigest(),
              'module_sha256':hashlib.sha256((runtime/'model.imasm').read_bytes()).hexdigest(),
              'source_input_record':str(parent/'execution_input.json'),
              'log':log,'checks':['complete tensor key set','executing head weights match emitted values exactly',
                                 'all other tensors retained exactly']}
    (output/'checkpoint_receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--bits',type=int,choices=range(16),default=3)
    parser.add_argument('--steps',type=int,default=32)
    args=parser.parse_args()
    save(args.root.resolve(),args.runtime.resolve(),args.parent.resolve(),args.out.resolve(),args.bits,args.steps)
