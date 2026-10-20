#!/usr/bin/env python3
"""Learned dynamic carrier state transform; exact FSPLIT/FFUSE remain executable operators.

The model learns the local state transition HOLD / AREV on each arm, with one
parameter-tied kernel for all arm positions and split arities. This is a neural
operator on the carrier, not an object-name classifier or a native IMASM VM.
"""
import argparse, json, random
from pathlib import Path
import torch
from torch import nn
from dynamic_bridge import split, reverse, evaluate

class CarrierKernel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5,32), nn.Tanh(), nn.Linear(32,4))
    def forward(self, masks, flags):
        # shape (...,4) bits + (...,1) AREV flag
        return self.net(torch.cat([masks, flags],-1))

def bits(m): return [(m>>i)&1 for i in range(4)]
def mask_from_bits(b): return sum((1<<i) for i,v in enumerate(b) if v)
def masks_data(seed_masks):
    X=[]; Y=[]
    for seed in seed_masks:
        for arity in (2,3):
            for flags in __import__('itertools').product((0,1),repeat=arity):
                arms=split(seed,arity)
                for arm, flag in zip(arms,flags):
                    X.append(bits(arm)+[flag]);Y.append(bits(reverse(arm) if flag else arm))
    return torch.tensor(X,dtype=torch.float32),torch.tensor(Y,dtype=torch.float32)

def assess(model, seeds):
    model.eval();n=0; arms_ok=0;cases=0;fused_ok=0;identity_correct=0
    with torch.no_grad():
        for seed in seeds:
            for arity in (2,3):
                for flags in __import__('itertools').product((0,1), repeat=arity):
                    before=split(seed,arity)
                    input_tensor=torch.tensor([bits(m)+[f] for m,f in zip(before,flags)],dtype=torch.float32)
                    preds=(model(input_tensor[:,:4],input_tensor[:,4:])>0).int().tolist()
                    arm_masks=[mask_from_bits(p) for p in preds]
                    expected=[reverse(m) if f else m for m,f in zip(before,flags)]
                    arms_ok+=sum(x==y for x,y in zip(arm_masks,expected));n+=arity
                    fused=0
                    for arm in arm_masks:fused|=arm
                    oracle=evaluate(seed,arity,flags)
                    fused_ok+=(fused==oracle['fused_mask'])
                    identity_correct+=((fused==seed)==oracle['recovered'])
                    cases+=1
    return {'cases':cases,'arm_accuracy':round(arms_ok/n,6),'fused_exact':round(fused_ok/cases,6), 'recovery_judgment_accuracy':round(identity_correct/cases,6)}

def train(args):
    torch.set_num_threads(1);random.seed(args.seed);torch.manual_seed(args.seed)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    order=list(range(16));random.Random(42).shuffle(order)
    train_masks=order[:12];val_masks=order[12:14];test_masks=order[14:]
    X,Y=masks_data(train_masks)
    model=CarrierKernel();opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    criterion=nn.BCEWithLogitsLoss()
    best=-1;history=[]
    for epoch in range(1,args.epochs+1):
        model.train();indices=torch.randperm(len(X));epoch_loss=0.
        for idx in indices.split(64):
            opt.zero_grad();loss=criterion(model(X[idx,:4],X[idx,4:]),Y[idx]);loss.backward();opt.step()
            epoch_loss+=loss.item()*len(idx)
        val=assess(model,val_masks)
        row={'epoch':epoch,'train_loss':round(epoch_loss/len(X),6),'validation':val};history.append(row)
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs: print(json.dumps(row))
        if val['fused_exact'] > best:
            best=val['fused_exact']
            torch.save({'model_state':model.state_dict(),'architecture':'CarrierKernel/5-32-4',
                        'seed':args.seed,'train_seeds':train_masks,'val_seeds':val_masks,
                        'test_seeds':test_masks},out/'best_carrier.pt')
    ckpt=torch.load(out/'best_carrier.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(ckpt['model_state'])
    results={'parameters':sum(p.numel() for p in model.parameters()),
             'best_validation':assess(model,val_masks),'test':assess(model,test_masks),
             'full_truth_table':assess(model,range(16)),
             'not_native_imasm':True,'test_seeds':test_masks,'checkpoint':str(out/'best_carrier.pt')}
    (out/'metrics.json').write_text(json.dumps({'results':results,'history':history},indent=2))
    print('FINAL',json.dumps(results))

def main():
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--lr',type=float,default=.01);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--out',default='carrier_checkpoints');p.add_argument('--log-every',type=int,default=25)
    args=p.parse_args();train(args)
if __name__=='__main__':main()
