#!/usr/bin/env python3
"""Compare shared-operator optimization on the source gate against exact seed returns."""
import argparse,json,random,sys
from pathlib import Path
import torch
from safetensors.torch import load_file
from retained_conditioning import evaluate


def probe(root,checkpoint,out,epochs,optimizers=('sgd','adam')):
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    sys.path.insert(0,str(root/'bob3ect/reentry_dnn'))
    from dtransformer_torch import DialetheicTransformerTorch
    config=json.loads((checkpoint/'config.json').read_text());state=load_file(str(checkpoint/'model.safetensors'))
    results={}
    for name in optimizers:
        model=DialetheicTransformerTorch(**config).double();model.load_state_dict(state)
        optimizer=torch.optim.SGD(model.parameters(),lr=.001) if name=='sgd' else torch.optim.Adam(model.parameters(),lr=.001)
        history=[];rng=random.Random(73);gradient_norms={}
        def query(seed):return evaluate(model,[256,71,0,266,266+seed,1,0,257],[],context=16,kind=2,bits=seed)
        for epoch in range(epochs+1):
            predicted=[];losses=[]
            with torch.no_grad():
                for seed in range(16):
                    result=query(seed);losses.append(result['loss'])
                    predicted.append(sum(int(result['first_logits'][k%2][259+k//2]>0)<<k for k in range(4)))
            history.append({'epoch':epoch,'exact':sum(x==i for i,x in enumerate(predicted)),
                            'predicted':predicted,'mean_loss':sum(losses)/16})
            if epoch==epochs:break
            seeds=list(range(16));rng.shuffle(seeds)
            for seed in seeds:
                optimizer.zero_grad();result=query(seed)
                for loss in result['loss_windows']:loss.backward()
                for parameter in model.parameters():parameter.grad.div_(result['labels'])
                if not gradient_norms:
                    gradient_norms={k:float(p.grad.norm()) for k,p in model.named_parameters()}
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        results[name]={'history':history,'first_gradient_norms':gradient_norms,'updates':epochs*16,'learning_rate':.001}
        print(json.dumps({'optimizer':name,'initial':history[0],'final':history[-1]}),flush=True)
    report={'checkpoint':str(checkpoint),'seed_targets':'compiled source VINIT identity on all sixteen seeds',
            'execution':'independent Torch optimization diagnostic; main IMASM campaign unchanged','results':results}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','checkpoint','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--epochs',type=int,default=64)
    p.add_argument('--optimizers',nargs='+',choices=('sgd','adam'),default=['sgd','adam'])
    a=p.parse_args();probe(a.root.resolve(),a.checkpoint.resolve(),a.out.resolve(),a.epochs,a.optimizers)
