#!/usr/bin/env python3
"""Train tied-weight reentry on *real, exactly specified crystal coordinate operations*.

Operations are decode(encode(x)) (roundtrip), change one named coordinate,
and sequential composition of two changes. These are arithmetic on the
Induxion mixed-radix crystal, NOT evidence of IMASM VM execution.
"""
import argparse, json, math, random
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from grammar_dnn import SharedReentry,RADICES,read_records

STRIDES=(4320000,864000,216000,43200,14400,2880,960,240,48,12,4,1)
OPERATIONS=('roundtrip','set_one','set_two')

def digits_from_index(index):
    if not 0 <= index < 17280000: raise ValueError(index)
    out=[]
    for radix,stride in zip(RADICES,STRIDES):
        digit,index=divmod(index,stride)
        assert digit<radix
        out.append(digit)
    return out

def index_from_digits(digits):
    if len(digits)!=12 or any(not 0<=n<r for n,r in zip(digits,RADICES)):
        raise ValueError('invalid digits')
    return sum(n*s for n,s in zip(digits,STRIDES))

def apply_operation(values,mode,axes,assigned):
    values=list(values)
    assert mode in range(3)
    if mode>=1: values[axes[0]]=assigned[0]
    if mode>=2: values[axes[1]]=assigned[1]
    assert digits_from_index(index_from_digits(values))==values
    return values

class OpExamples(Dataset):
    def __init__(self,rows,seed=0,repeat=3):
        self.examples=[]
        rng=random.Random(seed)
        for row in rows:
            old=row['targets']['indices']
            assert len(old)==12
            for j in range(repeat):
                mode=j%3
                a=rng.randrange(12)
                b=(a+rng.randrange(1,12))%12
                newa=rng.randrange(RADICES[a]);newb=rng.randrange(RADICES[b])
                target=apply_operation(old,mode,(a,b),(newa,newb))
                self.examples.append((old,mode,a,b,newa,newb,target))
    def __len__(self):return len(self.examples)
    def __getitem__(self,i):
        old,mode,a,b,newa,newb,target=self.examples[i]
        # A full 12-slot address plus two updates; special none marker at index 5.
        vals=torch.tensor(old+[newa,newb],dtype=torch.long)
        axes=torch.tensor([a,b],dtype=torch.long)
        return vals,axes,torch.tensor(mode),torch.tensor(target,dtype=torch.long)

class CrystalReentry(nn.Module):
    def __init__(self,d=32,heads=4,passes=3,ff_mult=2,dropout=0.1):
        super().__init__()
        self.config=dict(d=d,heads=heads,passes=passes,ff_mult=ff_mult,dropout=dropout)
        self.slot_embed=nn.Embedding(14,d)
        self.axis_value=nn.ModuleList(nn.Embedding(r,d) for r in RADICES)
        self.update_value=nn.Embedding(5,d)
        self.update_axis=nn.Embedding(12,d)
        self.mode_embed=nn.Embedding(3,d)
        self.reentry=SharedReentry(d,heads,ff_mult,dropout)
        self.final=nn.LayerNorm(d)
        self.heads=nn.ModuleList(nn.Linear(d,r) for r in RADICES)
    def forward(self,vals,axes,mode):
        b=vals.size(0)
        x=torch.stack([self.axis_value[i](vals[:,i]) for i in range(12)],dim=1)
        updates=self.update_value(vals[:,12:14])+self.update_axis(axes)
        updates[:,0,:]=updates[:,0,:]+self.mode_embed(mode)
        x=torch.cat([x,updates],dim=1)
        x=x+self.slot_embed(torch.arange(14,device=vals.device))[None,:,:]
        valid=torch.ones(b,14,device=vals.device,dtype=torch.bool)
        for _ in range(self.config['passes']):x=self.reentry(x,valid)
        x=self.final(x[:,:12,:])
        return [head(x[:,i,:]) for i,head in enumerate(self.heads)]

def evaluate(model,loader,device):
    model.eval();n=0;match=0;whole=0;loss_sum=0.
    with torch.no_grad():
        for vals,axes,mode,target in loader:
            vals,axes,mode,target=(t.to(device) for t in (vals,axes,mode,target))
            logits=model(vals,axes,mode)
            loss=sum(F.cross_entropy(logits[i],target[:,i]) for i in range(12))/12
            pred=torch.stack([l.argmax(-1) for l in logits],dim=1)
            match+=(pred==target).sum().item();whole+=(pred==target).all(dim=1).sum().item()
            n+=len(target);loss_sum+=loss.item()*len(target)
    return dict(samples=n,axis_accuracy=round(match/(12*n),5),exact_tuple_accuracy=round(whole/n,5),loss=round(loss_sum/n,5))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='grammar_training');p.add_argument('--out',default='operation_checkpoints')
    p.add_argument('--epochs',type=int,default=12);p.add_argument('--batch',type=int,default=128)
    p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=2)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--dim',type=int,default=32);p.add_argument('--passes',type=int,default=3)
    args=p.parse_args();torch.set_num_threads(args.threads);torch.manual_seed(args.seed)
    data=Path(args.data)
    datasets=[OpExamples(read_records(data/f'{split}.jsonl'),args.seed+ix) for ix,split in enumerate(('train','validation','test'))]
    loaders=[DataLoader(ds,batch_size=args.batch,shuffle=i==0) for i,ds in enumerate(datasets)]
    model=CrystalReentry(args.dim,passes=args.passes).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    params=sum(p.numel() for p in model.parameters());print('MODEL '+json.dumps({'parameters':params,'tasks':OPERATIONS}),flush=True)
    best=math.inf;history=[]
    for ep in range(1,args.epochs+1):
        model.train();loss_sum=0.
        for vals,axes,mode,target in loaders[0]:
            vals,axes,mode,target=(t.to(args.device) for t in (vals,axes,mode,target))
            optimizer.zero_grad(set_to_none=True)
            logits=model(vals,axes,mode)
            loss=sum(F.cross_entropy(logits[i],target[:,i]) for i in range(12))/12
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
            loss_sum+=loss.item()*len(target)
        val=evaluate(model,loaders[1],args.device)
        row={'epoch':ep,'train_loss':round(loss_sum/len(datasets[0]),5),'validation':val}
        print(json.dumps(row),flush=True);history.append(row)
        if val['loss']<best:
            best=val['loss'];torch.save({'model':model.state_dict(),'config':model.config,'task':OPERATIONS},out/'best_ops.pt')
    ck=torch.load(out/'best_ops.pt',map_location=args.device,weights_only=True)
    model.load_state_dict(ck['model'])
    report={'parameters':params,'best_validation_loss':best,'test':evaluate(model,loaders[2],args.device),'history':history,
            'scope':'mixed-radix crystal coordinate operations'}
    (out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print('FINAL '+json.dumps(report['test']),flush=True)
if __name__=='__main__':main()
