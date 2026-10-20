#!/usr/bin/env python3
"""CPU pilot on connected corpus tasks and independent carrier evidence targets."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

BOS, SEP, EOS, STATE, LAYER = 256, 257, 258, 259, 260


def select(path, limit, seed):
    rng, pools, counts = random.Random(seed), defaultdict(list), defaultdict(int)
    with path.open(encoding='utf8') as handle:
        for line in handle:
            row = json.loads(line)
            task = row['task']
            counts[task] += 1
            if len(pools[task]) < limit:
                pools[task].append(row)
            else:
                slot = rng.randrange(counts[task])
                if slot < limit:
                    pools[task][slot] = row
    return dict(pools)


def text_batch(rows, length, rng):
    xs, ys = [], []
    for row in rows:
        prompt = [BOS] + list(row['prompt'].encode('utf8'))[-length//2:] + [SEP]
        target = list(row['target'].encode('utf8')) + [EOS]
        stream = prompt + target
        start = rng.randrange(max(1, len(stream) - length))
        x = stream[start:start+length]
        y = [stream[i+1] if i+1 >= len(prompt) else -100
             for i in range(start, min(start+length, len(stream)-1))]
        x = x[:len(y)]
        xs.append(x + [0] * (length-len(x)))
        ys.append(y + [-100] * (length-len(y)))
    return torch.tensor(xs), torch.tensor(ys)


def carrier_rows(arity, gate, split, fuse):
    rows = []
    for seed in range(16):
        for work in '⊣≻≺⋈⊤⊙⊥⊞⊡':
            for arm in range(arity):
                arms = list(split(seed, arity))
                arms[arm] = gate(work, arms[arm])
                after = fuse(arms)
                prompt = f'FSPLIT={arity}; seed={seed}; arm={arm}; work={work}; FFUSE'
                rows.append({'prompt': prompt, 'after_bits': after, 'arity': arity,
                             'seed': seed, 'arm': arm, 'work': work,
                             'evidence': 'reference_carrier_semantics'})
    return rows


def evidence_batch(rows):
    tokens = [[BOS] + list(row['prompt'].encode('utf8')) + [SEP] for row in rows]
    lengths = torch.tensor([len(t)-1 for t in tokens])
    width = max(map(len, tokens))
    x = torch.tensor([t + [0]*(width-len(t)) for t in tokens])
    targets = torch.tensor([[(r['after_bits'] >> i) & 1 for i in range(4)]
                            for r in rows], dtype=torch.float)
    return x, lengths, targets


def evidence_logits(model, rows):
    x, lengths, target = evidence_batch(rows)
    s, r = model(x)
    batch = torch.arange(len(rows))
    logits = torch.stack([s[batch, lengths, STATE], r[batch, lengths, STATE],
                          s[batch, lengths, LAYER], r[batch, lengths, LAYER]], dim=-1)
    return logits, target


def train(root, views, output, steps=240, seed=17):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    sys.path.insert(0, str(root / 'bob3ect/reentry_dnn'))
    sys.path.insert(0, str(root / 'VGI/sources/GRAMMAR_UNIFIED_RUN'))
    from dtransformer_torch import DialetheicTransformerTorch, export_to_python
    from carrier_transitions import gate, split, fuse
    pools = {s: select(views / f'{s}.jsonl', 192 if s == 'train' else 24, seed)
             for s in ('train', 'validation', 'test')}
    cfg = dict(vocab_size=261, d_model=16, n_heads=2, n_layers=1, d_ff=32,
               max_len=64, causal=True, residual='blend')
    model = DialetheicTransformerTorch(**cfg)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=0.002)
    train_carriers = carrier_rows(2, gate, split, fuse)
    test_carriers = carrier_rows(3, gate, split, fuse)
    task_names = sorted(pools['train'])
    for split_name, tasks in pools.items():
        with (output / f'{split_name}_selection.jsonl').open('w', encoding='utf8') as handle:
            for rows in tasks.values():
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')

    @torch.no_grad()
    def evaluate():
        model.eval()
        result = {}
        for split_name in ('validation', 'test'):
            result[split_name] = {}
            for task, rows in sorted(pools[split_name].items()):
                x, y = text_batch(rows, 64, random.Random(0))
                sup, _ = model(x)
                mask = y != -100
                result[split_name][task] = {
                    'support_token_cross_entropy': float(F.cross_entropy(sup.reshape(-1, 261), y.reshape(-1))),
                    'next_byte_accuracy': float((sup.argmax(-1)[mask] == y[mask]).float().mean()),
                    'examples': len(rows)}
        for name, rows in [('seen_two_arm', train_carriers), ('heldout_three_arm', test_carriers)]:
            losses, exact, n = 0., 0, 0
            for start in range(0, len(rows), 32):
                logits, target = evidence_logits(model, rows[start:start+32])
                losses += float(F.binary_cross_entropy_with_logits(logits, target, reduction='sum'))
                exact += int(((logits > 0) == target.bool()).all(-1).sum())
                n += len(target)
            result[name] = {'four_bit_exact': exact/n, 'exact_cases': exact,
                            'cases': n, 'bit_cross_entropy': losses/(n*4)}
        return result

    baseline = evaluate()
    trace = []
    started = time.monotonic()
    for step in range(steps):
        model.train()
        task = task_names[step % len(task_names)]
        selected = rng.choices(pools['train'][task], k=12)
        x, y = text_batch(selected, 64, rng)
        sup, _ = model(x)
        text_loss = F.cross_entropy(sup.reshape(-1, 261), y.reshape(-1))
        logits, targets = evidence_logits(model, rng.choices(train_carriers, k=16))
        # Supports and refutations are independently supervised. Both and Neither
        # are ordinary targets; text ranking adds no refutation labels.
        evidence_loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss = text_loss + evidence_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step % 40 == 0 or step == steps-1:
            row = {'step': step+1, 'task': task, 'text_loss': float(text_loss.detach()),
                   'evidence_loss': float(evidence_loss.detach())}
            trace.append(row)
            print(json.dumps(row), flush=True)
    elapsed = time.monotonic() - started
    after = evaluate()
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
              str(output / 'model.safetensors'), metadata={'architecture': 'dialetheic_transformer',
                  'reentry': 'shared_final_trained_block'})
    (output / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    tokenizer = {'type': 'utf8_bytes', 'byte_ids': [0, 255], 'special_tokens':
                 {'BOS': BOS, 'SEP': SEP, 'EOS': EOS, 'STATE_TF': STATE, 'STATE_tf': LAYER},
                 'text_refutation_targets': 'unlabelled', 'max_context': 64}
    (output / 'tokenizer.json').write_text(json.dumps(tokenizer, indent=2) + '\n')
    # Bind a concrete held-out input to both execution presentations.
    example = pools['test']['catalog_to_tuple'][0]
    tokens = [BOS] + list(example['prompt'].encode('utf8'))[-6:] + [SEP]
    model = model.double().eval()
    py = export_to_python(model)
    with torch.no_grad():
        sup, ref = model(torch.tensor([tokens]))
    native = py.forward(tokens)
    residual = max(abs(a-b) for predicted, observed in
                   [(sup[0].tolist(), native.support), (ref[0].tolist(), native.refute)]
                   for aa, bb in zip(predicted, observed) for a, b in zip(aa, bb))
    assert residual < 1e-6, residual
    settled = py.forward(tokens, settle=True)
    (output / 'execution_input.json').write_text(json.dumps({'source_id': example['source_id'],
        'tokens': tokens, 'support': native.support, 'refute': native.refute}, indent=2) + '\n')
    report = {'seed': seed, 'steps': steps, 'device': 'cpu', 'parameters': model.num_parameters(),
              'training_seconds': elapsed, 'before': baseline, 'after': after, 'trace': trace,
              'export_max_abs_error': residual, 'reentry_kind': settled.settlement.kind,
              'reentry_cycle_length': settled.settlement.cycle_length,
              'evidence_domain': 'reference split/work/fuse; two-arm training, three-arm holdout',
              'task_scope': 'sampled byte windows; full construction and distinction policy remain under development',
              'views_report_sha256': hashlib.sha256((views / 'report.json').read_bytes()).hexdigest()}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'training_seconds': elapsed, 'export_max_abs_error': residual,
                      'reentry': report['reentry_kind'], 'after': after}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--views', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=240)
    args = parser.parse_args()
    train(args.root.resolve(), args.views, args.out, args.steps)
