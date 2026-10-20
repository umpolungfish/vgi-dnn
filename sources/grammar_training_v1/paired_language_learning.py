"""Independent source-backed language continuations through both DT readouts."""
import hashlib
import torch
import torch.nn.functional as F

from retained_conditioning import evaluate


class Readout:
    """Select a language readout while preserving the paired executing carrier."""
    def __init__(self, model, channel):
        self.model, self.channel = model, channel

    def __getattr__(self, name):
        return getattr(self.model, name)

    def head(self, t, f):
        support, refutation = self.model.head(t, f)
        return (support, refutation) if self.channel == 0 else (refutation, support)


def evaluate_record(model, row):
    if row['task'] != 'paired_language_evidence' or row['kind'] != 6:
        raise ValueError('paired language record required')
    prompt = row['prompt']
    if hashlib.sha256(prompt.encode()).hexdigest() != row['prompt_sha256']:
        raise ValueError('language prompt binding changed')
    if row['prefix'] != [256] + list(prompt.encode()) + [257]:
        raise ValueError('language source framing changed')
    losses, branches, labels = [], {}, 0
    for channel, name in enumerate(('support', 'refutation')):
        text = row[name + '_target']
        if (text is not None) != bool(row['evidence_bits'] & (1 << channel)):
            raise ValueError('language target presence disagrees with source evidence')
        if text is None:
            continue
        if hashlib.sha256(text.encode()).hexdigest() != row[name + '_sha256']:
            raise ValueError('language evidence binding changed')
        result = evaluate(Readout(model, channel), row['prefix'], list(text.encode()) + [258],
                          context=16, checkpointed=True)
        losses.append(sum(result['loss_windows']) / result['labels'])
        labels += result['labels']
        branches[name] = {'target': text, 'labels': result['labels'], 'loss': result['loss'],
                          'memory_t': result['memory'][0].detach().flatten().cpu().tolist(),
                          'memory_f': result['memory'][1].detach().flatten().cpu().tolist(),
                          'source_tokens_processed': result['source_tokens_processed']}
    if not losses and not row.get('evidence_presence'):
        raise ValueError('no attested language evidence to supervise')
    objective = sum(losses) / len(losses) if losses else model.E_t.new_zeros(())
    # A common source-only return accompanies both branch continuations.
    source = (evaluate(model, row['prefix'], [], context=16, kind=2, checkpointed=True, differentiable_state=True)
              if row.get('evidence_presence') else
              evaluate(model, row['prefix'], [258], context=16, checkpointed=True))
    presence = None
    if row.get('evidence_presence'):
        support, refutation = source['first_logits']
        # The constructive T/F coordinates of the existing sixteen-state readout.
        logits = torch.stack((support[259], refutation[259]))
        target = logits.new_tensor([row['evidence_bits'] & 1, (row['evidence_bits'] >> 1) & 1])
        observed_mask = row.get('presence_mask')
        if observed_mask is None:
            evidence_loss = F.binary_cross_entropy_with_logits(logits, target)
            observed_count = 2
        else:
            if len(observed_mask) != 2 or any(v not in (0,1) for v in observed_mask) or not any(observed_mask):
                raise ValueError('invalid observed evidence mask')
            mask = logits.new_tensor(observed_mask)
            evidence_loss = (F.binary_cross_entropy_with_logits(logits,target,reduction='none')*mask).sum()/mask.sum()
            observed_count = sum(observed_mask)
        objective = objective + evidence_loss
        labels += observed_count
        presence = {'target_bits': row['evidence_bits'],
                    'proposal_bits': int(logits[0] > 0) | (int(logits[1] > 0) << 1),
                    'logits': logits.detach().cpu().tolist(), 'loss': float(evidence_loss.detach())}
        if observed_mask is not None:
            presence['observed_mask'] = observed_mask
    result = {'loss_windows': [objective * labels], 'labels': labels,
            'loss': float(objective.detach()), 'memory': source['memory'],
            'source_tokens_processed': sum(b['source_tokens_processed'] for b in branches.values()) + (source['source_tokens_processed'] if presence is not None else 0),
            'language_branches': branches, 'objective': 'independent_support_refutation_language'}
    if presence is not None:
        result['language_presence'] = presence
    return result
