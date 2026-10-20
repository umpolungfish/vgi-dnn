"""Retain independent action support and refutation in shared graph updates."""
import ctypes
import hashlib
from pathlib import Path

from distinction_format import tokens, words, ACTIONS
from retained_conditioning import evaluate


def prepare(rows, library):
    oracle = library.vgi_retained_oracle
    oracle.argtypes = [ctypes.c_ulong, ctypes.c_ulong, ctypes.c_long, ctypes.c_long]
    oracle.restype = ctypes.c_long
    observe = library.vgi_retained_oracle_observe
    observe.argtypes = [ctypes.c_long]
    observe.restype = ctypes.c_long
    checked = set()
    pool = []
    for row in rows:
        if row['kind'] != 3:
            continue
        reference = row['reference']
        if row['training_split'] != 'train' or reference['split'] != 'train':
            raise ValueError('distinction rehearsal requires training records')
        source = row['source']
        key = (source['path'], source['sha256'])
        if key not in checked:
            if hashlib.sha256(Path(key[0]).read_bytes()).hexdigest() != key[1]:
                raise ValueError('distinction source changed')
            checked.add(key)
        if tokens(reference) != row['prefix'] or row['target']:
            raise ValueError('distinction query binding changed')
        first, second = words(row['prefix'])
        exact = int(oracle(first, second, 0, 0))
        expected = sum((reference['support'][a] << (2*a)) |
                       (reference['refute'][a] << (2*a+1)) for a in range(4))
        if exact != expected or not 0 <= exact < 256:
            raise ValueError('executing distinction oracle disagrees with source evidence')
        pool.append({'record': row, 'target_bits': exact,
                     'candidate_returns': [int(observe(20+a)) for a in range(4)]})
    if not pool:
        raise ValueError('no oracle-bound distinction records')
    return pool


def loss(model, pool, index, count, weight):
    total = None
    observations = []
    for offset in range(count):
        source = pool[(index * count + offset) % len(pool)]
        row, bits = source['record'], source['target_bits']
        result = evaluate(model, row['prefix'], [], context=16, kind=1, bits=bits)
        value = sum(result['loss_windows']) / result['labels']
        total = value if total is None else total + value
        support, refutation = result['first_logits']
        proposal = sum((int(support[action] > 0) << (2*a)) |
                       (int(refutation[action] > 0) << (2*a+1)) for a, action in enumerate(ACTIONS))
        observations.append({'example_id': row['example_id'], 'source_id': row['source_id'],
                             'query': row['prefix'], 'target_bits': bits, 'proposal_bits': proposal,
                             'candidate_returns': source['candidate_returns'],
                             'target_states': [(bits >> (2*a)) & 3 for a in range(4)],
                             'proposal_states': [(proposal >> (2*a)) & 3 for a in range(4)],
                             'loss': float(value.detach())})
    return weight * total / count, observations
