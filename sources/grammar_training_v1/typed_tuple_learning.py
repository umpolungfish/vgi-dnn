"""Twelve dependent finite choices through the existing retained learner.

UTF-8 is only the transport encoding for each admissible value. The learned
decision and its normalizer range over the current axis, never over byte IDs.
No supplied tuple is consulted by infer(); attestations supervise objective().
"""
import torch
import torch.nn.functional as F

from bootstrap import AXES, encode_address
from retained_conditioning import evaluate


def choices(model, prompt, previous, axis):
    if not 0 <= axis < len(AXES) or len(previous) != axis:
        raise ValueError('axis and carried coordinate prefix disagree')
    for glyph, (_, _, alphabet) in zip(previous, AXES):
        if glyph not in alphabet:
            raise ValueError('invalid carried coordinate')
    symbol, name, alphabet = AXES[axis]
    query = (prompt + '\nTyped coordinates: ' + previous +
             '\nAxis ' + symbol + ' (' + name + '): ')
    prefix = [256] + list(query.encode('utf8')) + [257]
    # Retain the complete query and its differentiable memory once. Candidate
    # branches share that source state; each consumes one whole allowed glyph.
    source = evaluate(model, prefix, [258], differentiable_state=True,
                      checkpointed=True)
    memory = source['memory']
    scores = []
    for glyph in alphabet:
        branch = evaluate(model, [257], list(glyph.encode('utf8')),
                          initial_memory=memory, differentiable_state=True)
        scores.append(-sum(branch['loss_windows']))
    return torch.stack(scores), source


def objective(model, prompt, address):
    encode_address(address)  # reject malformed training evidence before execution
    terms, observations = [], []
    for axis, glyph in enumerate(address):
        scores, source = choices(model, prompt, address[:axis], axis)
        alphabet = AXES[axis][2]
        target = alphabet.index(glyph)
        terms.append(F.cross_entropy(scores[None], scores.new_tensor([target], dtype=torch.long)))
        observations.append({'axis': AXES[axis][0], 'allowed': list(alphabet),
                             'target': glyph, 'proposal': alphabet[int(scores.detach().argmax())],
                             'scores': scores.detach().cpu().tolist(),
                             'source_tokens_processed': source['source_tokens_processed'],
                             'memory_t': source['memory'][0].detach().flatten().cpu().tolist(),
                             'memory_f': source['memory'][1].detach().flatten().cpu().tolist()})
    return torch.stack(terms).mean(), observations


def evaluate_record(model, row):
    """Adapt a source-bound retained tuple record to the shared objective API."""
    import hashlib
    if row['task'] not in ('catalog_to_tuple', 'ob3ect_to_tuple'):
        raise ValueError('typed tuple task required')
    if row['prefix'][0] != 256 or row['prefix'][-1] != 257 or row['target'][-1] != 258:
        raise ValueError('invalid retained tuple framing')
    prompt = bytes(row['prefix'][1:-1]).decode('utf8')
    address = bytes(row['target'][:-1]).decode('utf8')
    for field, value in [('prompt_sha256', prompt), ('target_sha256', address)]:
        if hashlib.sha256(value.encode()).hexdigest() != row[field]:
            raise ValueError('typed tuple source binding mismatch')
    loss, observations = objective(model, prompt, address)
    last = observations[-1]
    memory = tuple(torch.tensor(last[key], dtype=model.E_t.dtype, device=model.E_t.device).reshape(1, 1, -1)
                   for key in ('memory_t', 'memory_f'))
    return {'loss_windows': [loss * 12], 'labels': 12, 'loss': float(loss.detach()),
            'memory': memory, 'source_tokens_processed': sum(o['source_tokens_processed'] for o in observations),
            'typed_observations': observations, 'objective': 'twelve_axis_choices'}


@torch.no_grad()
def infer(model, prompt):
    address, observations = '', []
    for axis, (symbol, _, alphabet) in enumerate(AXES):
        scores, source = choices(model, prompt, address, axis)
        selected = alphabet[int(scores.argmax())]
        observations.append({'axis': symbol, 'allowed': list(alphabet),
                             'selected': selected, 'scores': scores.cpu().tolist(),
                             'source_tokens_processed': source['source_tokens_processed']})
        address += selected
    return {'address': address, 'crystal_index': encode_address(address),
            'observations': observations}
