"""Reference retained DT: all source tokens enter a shared recurrent operation."""
import torch
import torch.nn.functional as F


def _chunk(model, mt, mf, ids, position, count, active, labels, kind, unroll):
    from dtransformer_torch import logit, _residual
    t = torch.sigmoid(model.E_t[ids] + .5 * model.pe[position])
    f = torch.sigmoid(model.E_f[ids] + .5 * model.pe[(position + 7) % model.max_len])
    t, f = torch.cat([mt, t], 1), torch.cat([mf, f], 1)
    for block in model.blocks:
        t, f = block(t, f)
    for _ in range(unroll):
        t, f = model.reentry_block(t, f)
    t, f = _residual((mt.expand_as(t), mf.expand_as(f)), (t, f),
                     model.reentry_block.gate1, model.cfg['residual'])
    if active:
        nt, nf = model.final_norm(t[:, active], f[:, active])
        support, refutation = model.head(logit(nt), logit(nf))
        if kind == 0:
            loss = F.cross_entropy(support[0], labels, reduction='sum')
        else:
            width, base = (8, 261) if kind == 1 else (2, 265) if kind == 3 else (4, 259)
            logits = torch.stack([(refutation if k % 2 else support)[0, -1, base + k // 2] for k in range(width)])
            loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='sum')
        first_t, first_f = support[0, 0], refutation[0, 0]
    else:
        loss = t.new_zeros(())
        first_t = first_f = t.new_empty((0,))
    return t[:, count:count + 1], f[:, count:count + 1], loss, first_t, first_f


def evaluate(model, prefix, target, context=16, bptt=0, kind=0, bits=0, unroll=0,
             initial_memory=None, differentiable_state=False, checkpointed=False):
    if context < 2 or context > model.max_len:
        raise ValueError('invalid retained context')
    if not prefix or (kind == 0 and not target):
        raise ValueError('missing source or target')
    sequence = prefix + target
    input_count = len(sequence) - 1 if kind == 0 else len(prefix)
    memory = (torch.full((1, 1, model.d), .05, dtype=model.E_t.dtype, device=model.E_t.device),) * 2
    if initial_memory is not None:
        memory = tuple(torch.as_tensor(v, dtype=model.E_t.dtype, device=model.E_t.device).reshape(1, 1, model.d) for v in initial_memory)
        if not differentiable_state:
            memory = tuple(v.detach() for v in memory)
    losses = []; window_loss = None; label_count = 0; chunks = 0; first = None; trajectory = []
    for offset in range(0, input_count, context - 1):
        count = min(context - 1, input_count - offset)
        values = sequence[offset:offset + count] + [0] * (context - 1 - count)
        ids = torch.tensor([values], device=model.E_t.device)
        position = torch.arange(offset, offset + context - 1, device=model.E_t.device) % model.max_len
        active = [i for i in range(1, count + 1) if
                  (offset + i >= len(prefix) if kind == 0 else offset + i == input_count)]
        width = 8 if kind == 1 else 2 if kind == 3 else 4
        labels = (torch.tensor([sequence[offset + i] for i in active], device=model.E_t.device) if kind == 0
                  else torch.tensor([(bits >> k) & 1 for k in range(width)], dtype=model.E_t.dtype, device=model.E_t.device))
        arguments = (model, *memory, ids, position, count, active, labels, kind, unroll)
        if checkpointed and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            mt, mf, loss, first_t, first_f = checkpoint(_chunk, *arguments, use_reentrant=False)
        else:
            mt, mf, loss, first_t, first_f = _chunk(*arguments)
        memory = mt, mf
        if active:
            if first is None:
                first = (first_t, first_f) if differentiable_state else (first_t.detach(), first_f.detach())
            label_count += len(active) if kind == 0 else width
            window_loss = loss if window_loss is None else window_loss + loss
        chunks += 1
        trajectory.append({'source_end': offset + count, 'memory_t': mt.detach().tolist(), 'memory_f': mf.detach().tolist()})
        if (bptt and chunks % bptt == 0) or offset + count == input_count:
            if window_loss is not None:
                losses.append(window_loss); window_loss = None
            if not (differentiable_state and offset + count == input_count):
                memory = memory[0].detach(), memory[1].detach()
    return {'loss_windows': losses, 'labels': label_count,
            'loss': sum(float(loss.detach()) for loss in losses) / label_count,
            'memory': memory, 'first_logits': first, 'trajectory': trajectory,
            'source_tokens_processed': input_count}
