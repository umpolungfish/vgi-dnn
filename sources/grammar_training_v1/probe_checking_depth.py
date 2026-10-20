#!/usr/bin/env python3
"""Compare repeated shared-block evaluation on the same checking curriculum."""
import argparse,json,random,sys
from collections import defaultdict
from pathlib import Path
import torch
from optimizer_state import load_model
from retained_conditioning import evaluate
from torch_reference_optimizer import make_optimizer


def probe(root,runtime,training,tasks,out,epochs,depths):
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    sys.path.insert(0,str(root/'bob3ect/reentry_dnn'))
    from dtransformer_torch import DialetheicTransformerTorch
    build=json.loads((runtime/'build.json').read_text());initial=load_model(Path(build['parent'])/'model.safetensors')
    train=[json.loads(l) for l in training.open()];groups=defaultdict(list)
    for line in (tasks/'validation.jsonl').open():
        row=json.loads(line);groups[row['reference_bits']].append(row)
    rng=random.Random(79);held=[]
    for bits in sorted(groups):rng.shuffle(groups[bits]);held.extend(groups[bits][:32])
    reports=[]
    for depth in depths:
        model=DialetheicTransformerTorch(**build['config']).double();model.load_state_dict(initial)
        optimizer=make_optimizer(model,build);history=[];updates=0
        for epoch in range(epochs+1):
            exact=calls=correct_calls=wrong_unchecked=0
            with torch.no_grad():
                for row in held:
                    result=evaluate(model,row['prefix'],[],context=16,kind=3,bits=0,unroll=depth)
                    predicted=int(result['first_logits'][0][265]>0)|(int(result['first_logits'][1][265]>0)<<1)
                    exact+=predicted==row['reference_bits'];calls+=predicted&1
                    correct_calls+=(predicted&1)==(row['reference_bits']&1)
                    wrong_unchecked+=not(predicted&1) and row['checking_reference']['proposal']!=row['checking_reference']['carrier']
            history.append({'epoch':epoch,'updates':updates,'queries':len(held),'evidence_exact':exact,
                            'check_bit_exact':correct_calls,'oracle_choices':calls,'wrong_unchecked':wrong_unchecked})
            if epoch==epochs:break
            for row in train:
                optimizer.zero_grad();result=evaluate(model,row['prefix'],[],context=16,kind=3,bits=row['reference_bits'],unroll=depth)
                predicted=int(result['first_logits'][0][265]>0)|(int(result['first_logits'][1][265]>0)<<1)
                if predicted==row['reference_bits']:continue
                for loss in result['loss_windows']:loss.backward()
                for p in model.parameters():p.grad.div_(result['labels'])
                norm=float(torch.sqrt(sum((p.grad*p.grad).sum() for p in model.parameters())))
                scale=min(1.,build['stream_gradient_clip']/norm) if norm else 1.
                for p in model.parameters():p.grad.mul_(scale)
                optimizer.step();updates+=1
        reports.append({'additional_shared_block_passes':depth,'history':history})
        print(json.dumps(reports[-1]),flush=True)
    report={'comparison':'same initial checkpoint and source-bound curriculum; optional repetitions of the trained final block',
            'execution':'independent Torch diagnostic; executing campaign unchanged',
            'validation_subset':'32 deterministic queries per evidence shape; withheld from updates',
            'validation_ids':[r['example_id'] for r in held],'runtime':str(runtime),'reports':reports}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','runtime','training','tasks','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--epochs',type=int,default=2);p.add_argument('--depths',nargs='+',type=int,default=[0,2])
    a=p.parse_args();probe(a.root.resolve(),a.runtime.resolve(),a.training.resolve(),a.tasks.resolve(),a.out.resolve(),a.epochs,a.depths)
