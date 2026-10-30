#!/usr/bin/env python3
"""Trainable, weight-tied re-entry model for 12-axis typed imscription prediction.
This is a neural baseline aligned to the catalog's axes, NOT an IMASM VM or a
claim of SIXTEEN_3 execution/proof.  Target labels are catalog attestations.
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

AXES = ('⊢','⊣','≻','≺','⋈','⊤','∈','∋','⊙','⊥','⊞','⊡')
RADICES = (4,5,4,5,3,5,3,4,5,4,3,4)


def read_records(file):
    return [json.loads(s) for s in Path(file).read_text(encoding='utf8').splitlines() if s.strip()]


def make_vocab(rows):
    chars = sorted({c for row in rows for c in row['object_name']})
    return {'<PAD>':0,'<UNK>':1, **{c:i+2 for i,c in enumerate(chars)}}


def encode_name(name, vocab, max_len):
    return [vocab.get(c,1) for c in name[:max_len]]


class Names(Dataset):
    def __init__(self, rows, vocab, max_len):
        self.rows = rows
        self.vocab = vocab
        self.max_len = max_len
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        ids = encode_name(r['object_name'],self.vocab,self.max_len)
        return torch.tensor(ids + [0]*(self.max_len-len(ids)), dtype=torch.long), torch.tensor(r['targets']['indices'],dtype=torch.long)


class SharedReentry(nn.Module):
    def __init__(self, d, heads, ff_mult, dropout):
        super().__init__()
        self.attn=nn.MultiheadAttention(d,heads,dropout=dropout,batch_first=True)
        self.norm1=nn.LayerNorm(d)
        self.norm2=nn.LayerNorm(d)
        self.ff=nn.Sequential(nn.Linear(d,d*ff_mult),nn.GELU(),nn.Dropout(dropout),nn.Linear(d*ff_mult,d))
        self.dropout=nn.Dropout(dropout)
    def forward(self, x, valid):
        # Query positions always exist; only padded name characters are ignored.
        q=self.norm1(x)
        z,_=self.attn(q,q,q,key_padding_mask=~valid,need_weights=False)
        x=x+self.dropout(z)
        return x+self.dropout(self.ff(self.norm2(x)))


class GrammarDNN(nn.Module):
    def __init__(self, vocab_size, d=32, heads=4, passes=3, ff_mult=2, dropout=.1, max_len=64):
        super().__init__()
        self.config=dict(vocab_size=vocab_size,d=d,heads=heads,passes=passes,ff_mult=ff_mult,dropout=dropout,max_len=max_len)
        self.embedding=nn.Embedding(vocab_size,d,padding_idx=0)
        self.pos=nn.Embedding(max_len,d)
        self.axis_queries=nn.Parameter(torch.randn(12,d)*0.02)
        self.reentry=SharedReentry(d,heads,ff_mult,dropout)
        self.final=nn.LayerNorm(d)
        self.heads=nn.ModuleList(nn.Linear(d,r) for r in RADICES)
    def forward(self, tokens):
        b,n=tokens.shape
        pos=self.pos(torch.arange(n,device=tokens.device))[None,:,:]
        names=self.embedding(tokens)+pos
        queries=self.axis_queries.unsqueeze(0).expand(b,-1,-1)
        x=torch.cat([queries,names],dim=1)
        valid=torch.cat([torch.ones((b,12),device=tokens.device,dtype=torch.bool),tokens!=0],dim=1)
        # Same learned transformation is re-entered, with tied weights on every pass.
        for _ in range(self.config['passes']):
            x=self.reentry(x,valid)
        x=self.final(x[:,:12,:])
        return [head(x[:,i,:]) for i,head in enumerate(self.heads)]


def loss_logits(logits, target):
    return sum(F.cross_entropy(pred,target[:,i]) for i,pred in enumerate(logits))/12


def metrics(logits, target):
    preds=torch.stack([v.argmax(-1) for v in logits],dim=-1)
    matches=(preds==target)
    return matches.sum().item(),matches.numel(),matches.all(-1).sum().item(),target.shape[0]


def evaluate(model,loader,device):
    model.eval()
    totals=[0,0,0,0]
    ls=0.
    with torch.no_grad():
        for inp,y in loader:
            inp,y=inp.to(device),y.to(device)
            logits=model(inp)
            ls+=loss_logits(logits,y).item()*len(y)
            totals=[a+b for a,b in zip(totals,metrics(logits,y))]
    return {'loss':round(ls/max(totals[3],1),5), 'axis_accuracy':round(totals[0]/max(totals[1],1),5), 'exact_tuple_accuracy':round(totals[2]/max(totals[3],1),5), 'samples':totals[3]}


def train(a):
    random.seed(a.seed);torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    folder=Path(a.data)
    train_rows=read_records(folder/'train.jsonl')
    val_rows=read_records(folder/'validation.jsonl')
    test_rows=read_records(folder/'test.jsonl')
    assert all(len(r['targets']['indices'])==12 for r in train_rows+val_rows+test_rows)
    addr_sets=[{r['address'] for r in rows} for rows in (train_rows,val_rows,test_rows)]
    assert all(not addr_sets[i]&addr_sets[j] for i in range(3) for j in range(i+1,3)), 'address split leakage'
    vocab=make_vocab(train_rows)
    loaders=[DataLoader(Names(rows,vocab,a.max_len),batch_size=a.batch,shuffle=(i==0),num_workers=0)
             for i,rows in enumerate((train_rows,val_rows,test_rows))]
    device=torch.device(a.device)
    model=GrammarDNN(len(vocab),a.dim,a.heads,a.passes,a.ff_mult,a.dropout,a.max_len).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    report={'configuration':vars(a),'parameters':sum(p.numel() for p in model.parameters()),'epochs':[]}; print('MODEL '+json.dumps({'parameters':report['parameters'],'dim':a.dim,'passes':a.passes}),flush=True)
    best=math.inf
    for epoch in range(1,a.epochs+1):
        model.train()
        trainloss=0.
        for tokens,y in loaders[0]:
            tokens,y=tokens.to(device),y.to(device)
            opt.zero_grad(set_to_none=True)
            loss=loss_logits(model(tokens),y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
            trainloss+=loss.item()*len(y)
        v=evaluate(model,loaders[1],device)
        entry={'epoch':epoch,'training_loss':round(trainloss/len(train_rows),5),'validation':v}
        report['epochs'].append(entry)
        print(json.dumps(entry),flush=True)
        if v['loss']<best:
            best=v['loss']
            torch.save({'state_dict':model.state_dict(),'config':model.config,'vocab':vocab,
                        'axis_alphabets':[x['alphabet'] for x in json.loads((folder/'schema.json').read_text(encoding='utf8'))['axes']],
                        'training_note':'catalog typed-address prediction; no operational IMASM proof or SIXTEEN_3 transition'},out/'best.pt')
    checkpoint=torch.load(out/'best.pt',map_location=device,weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    report['best_validation_loss']=best
    report['test']=evaluate(model,loaders[2],device)
    (out/'metrics.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
    print('FINAL '+json.dumps({'parameters':report['parameters'],'test':report['test'],'checkpoint':str(out/'best.pt')}),flush=True)


def predict(a):
    torch.set_num_threads(a.threads)
    ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    model=GrammarDNN(**ckpt['config']);model.load_state_dict(ckpt['state_dict']);model.eval()
    ids=encode_name(a.name,ckpt['vocab'],ckpt['config']['max_len'])
    x=torch.tensor([ids+[0]*(ckpt['config']['max_len']-len(ids))])
    with torch.no_grad(): logits=model(x)
    indices=[l[0].argmax().item() for l in logits]
    glyphs=[ckpt['axis_alphabets'][i][v] for i,v in enumerate(indices)]
    confidence=[round(F.softmax(l[0],dim=-1).max().item(),4) for l in logits]
    print(json.dumps({'object_name':a.name,'predicted_address':'⟨'+''.join(glyphs)+'⟩',
                      'indices':indices,'axis_confidence':confidence,
                      'warning':'learned catalog-coordinate prediction'},ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser(description='Weight-tied re-entry Grammar DNN, typed 12-axis imscription')
    sub=p.add_subparsers(dest='cmd',required=True)
    t=sub.add_parser('train')
    t.add_argument('--data',default='grammar_training');t.add_argument('--out',default='checkpoints')
    t.add_argument('--epochs',type=int,default=12);t.add_argument('--batch',type=int,default=64)
    t.add_argument('--dim',type=int,default=32);t.add_argument('--heads',type=int,default=4)
    t.add_argument('--passes',type=int,default=3);t.add_argument('--ff-mult',type=int,default=2)
    t.add_argument('--dropout',type=float,default=.1);t.add_argument('--lr',type=float,default=.001)
    t.add_argument('--weight-decay',type=float,default=.01);t.add_argument('--max-len',type=int,default=64)
    t.add_argument('--seed',type=int,default=42);t.add_argument('--threads',type=int,default=2)
    t.add_argument('--device',default='cpu')
    i=sub.add_parser('predict');i.add_argument('name');i.add_argument('--checkpoint',default='checkpoints/best.pt');i.add_argument('--threads',type=int,default=2)
    a=p.parse_args()
    if a.cmd=='train':train(a)
    else:predict(a)
if __name__=='__main__':main()
