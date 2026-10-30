#!/usr/bin/env python3
"""Generate independent language branches selected by learned paired evidence."""
import argparse
import json
from pathlib import Path
import sys

import torch
from optimizer_state import load_model
from paired_language_learning import Readout
from retained_conditioning import evaluate
from train_cuda_trajectories import sha


@torch.no_grad()
def generate(root, checkpoint, prompts, out, max_new=96, resume=None):
    torch.set_num_threads(1)
    sys.path.insert(0, str(root / 'bob3ect/reentry_dnn'))
    from dtransformer_torch import DialetheicTransformerTorch
    model = DialetheicTransformerTorch(**json.loads((checkpoint/'config.json').read_text())).double()
    model.load_state_dict(load_model(checkpoint/'model.safetensors'))
    model.eval()
    binding = sha(checkpoint/'model.safetensors')
    state = json.loads(resume.read_text()) if resume else {'checkpoint_sha256':binding,'context_bytes':[],'turns':[]}
    if state['checkpoint_sha256'] != binding:
        raise ValueError('conversation belongs to a different returned model')
    context = state['context_bytes']
    for prompt in prompts:
        formatted = prompt if prompt.startswith('User:') else 'User: ' + prompt + '\nAssistant:'
        source = context + list(formatted.encode())
        prefix = [256] + source + [257]
        result = evaluate(model,prefix,[],context=16,kind=2)
        support,refutation = result['first_logits']
        logits = [float(support[259]),float(refutation[259])]
        evidence = int(logits[0]>0) | (int(logits[1]>0)<<1)
        turn = {'prompt':prompt,'evidence_bits':evidence,'evidence_logits':logits,
                'source_bank':prefix,'memory_t':result['memory'][0].flatten().tolist(),
                'memory_f':result['memory'][1].flatten().tolist(),'branches':{}}
        replies = []
        for channel,name in enumerate(('support','refutation')):
            if not evidence & (1<<channel):
                turn['branches'][name] = None
                continue
            readout = Readout(model,channel)
            generated,trace,ended = [],[],False
            for _ in range(max_new):
                value = evaluate(readout,prefix+generated,[258],context=16)
                current,other = value['first_logits']
                allowed = torch.tensor(list(range(256))+[258])
                token = int(allowed[current[allowed].argmax()])
                trace.append({'token':token,'selected_readout_logit':float(current[token]),
                              'paired_readout_logit':float(other[token])})
                if token == 258:
                    ended=True
                    break
                generated.append(token)
            final = evaluate(readout,prefix+generated,[258],context=16)
            turn['branches'][name] = {'text':bytes(generated).decode('utf8',errors='replace'),
                                     'generated_bytes':generated,'ended_with_eos':ended,'trace':trace,
                                     'source_bank':prefix+generated+([258] if ended else []),
                                     'memory_t':final['memory'][0].flatten().tolist(),
                                     'memory_f':final['memory'][1].flatten().tolist()}
            replies += list(('\n'+name+': ').encode()) + generated
        context = source + replies + [10]
        state['turns'].append(turn)
        print(json.dumps({'evidence_bits':evidence,'branches':{k:v['text'] if v else None for k,v in turn['branches'].items()}},ensure_ascii=False),flush=True)
    state.update(context_bytes=context,checkpoint=str(checkpoint),
                 execution='learned constructive T/F presence and independently generated language readouts; both branches retained through conversation re-entry')
    out.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','checkpoint','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--prompt',action='append',required=True)
    parser.add_argument('--max-new-bytes',type=int,default=96)
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    if args.max_new_bytes<1:
        parser.error('positive generation length required')
    generate(args.root.resolve(),args.checkpoint.resolve(),args.prompt,args.out.resolve(),args.max_new_bytes,args.resume)
