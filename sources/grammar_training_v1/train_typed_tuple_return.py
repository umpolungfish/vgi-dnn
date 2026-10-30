#!/usr/bin/env python3
"""Return a provenance-bound typed-coordinate update with existing Adam state."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file

from optimizer_state import split_state
from train_cuda_trajectories import adam_step
from typed_tuple_learning import objective, infer


def train(root, checkpoint, build, records, out, count=1):
    torch.set_num_threads(1)
    sys.path.insert(0, str(root / 'bob3ect/reentry_dnn'))
    from dtransformer_torch import DialetheicTransformerTorch
    metadata = json.loads(build.read_text())
    weights, optimizer = split_state(load_file(str(checkpoint / 'model.safetensors')))
    model = DialetheicTransformerTorch(**metadata['config']).double()
    model.load_state_dict(weights)
    rows = [json.loads(line) for line in records.read_text().splitlines()]
    rows = [r for r in rows if r.get('task') in ('catalog_to_tuple', 'ob3ect_to_tuple')]
    if count < 1 or len(rows) < count:
        raise ValueError('insufficient typed training records')
    selected = rows[:count]
    if any(r.get('split', r.get('training_split', 'train')) != 'train' for r in selected):
        raise ValueError('typed updates require training partition')
    out.mkdir(parents=True, exist_ok=False)
    returned = out / 'returned_checkpoint'
    shutil.copytree(checkpoint, returned)
    observations = []
    initial_clock = int(optimizer['clock'][0])
    for row in selected:
        # Prepared retained rows keep the original source record under source.
        source = row.get('source', row)
        prompt = source.get('prompt', row.get('prompt'))
        address = source.get('target', row.get('target'))
        if prompt is None and 'prefix' in row:
            if row['prefix'][0] != 256 or row['prefix'][-1] != 257:
                raise ValueError('invalid source framing')
            prompt = bytes(row['prefix'][1:-1]).decode('utf8')
        if isinstance(address, list):
            if address[-1] != 258:
                raise ValueError('invalid target framing')
            address = bytes(address[:-1]).decode('utf8')
        if not isinstance(prompt, str) or not isinstance(address, str):
            raise ValueError('complete source prompt and attested tuple required')
        for field, value in (('prompt_sha256', prompt), ('target_sha256', address)):
            if field in row and hashlib.sha256(value.encode('utf8')).hexdigest() != row[field]:
                raise ValueError('typed source binding mismatch: ' + field)
        before = infer(model, prompt)
        model.zero_grad(set_to_none=True)
        loss, evidence = objective(model, prompt, address)
        loss.backward()
        norm = adam_step(model, optimizer, metadata['parameters'], .0001, 1.)
        with torch.no_grad():
            after_loss, _ = objective(model, prompt, address)
        after = infer(model, prompt)
        observations.append({'source': row, 'before': before, 'after': after,
                             'loss_before': float(loss.detach()),
                             'loss_after': float(after_loss), 'gradient_norm': norm,
                             'axis_evidence': evidence})
    state = {k: v.detach().contiguous() for k, v in model.state_dict().items()}
    state.update({'__optimizer.' + k: v.contiguous() for k, v in optimizer.items()})
    save_file(state, str(returned / 'model.safetensors'))
    report = {'parent': str(checkpoint), 'parent_sha256': hashlib.sha256((checkpoint / 'model.safetensors').read_bytes()).hexdigest(),
              'records': str(records), 'records_sha256': hashlib.sha256(records.read_bytes()).hexdigest(),
              'implementation_sha256': hashlib.sha256(Path(__file__).with_name('typed_tuple_learning.py').read_bytes()).hexdigest(),
              'initial_clock': initial_clock, 'returned_clock': int(optimizer['clock'][0]),
              'observations': observations,
              'scope': 'CPU typed-coordinate learning branch; pending native/Vox and production integration'}
    (returned / 'typed_tuple_state.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'returned': str(returned), 'updates': len(observations),
                      'losses': [[r['loss_before'], r['loss_after']] for r in observations]}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'checkpoint', 'build', 'records', 'out'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--count', type=int, default=1)
    a = p.parse_args()
    train(a.root.resolve(), a.checkpoint.resolve(), a.build.resolve(), a.records.resolve(), a.out.resolve(), a.count)
