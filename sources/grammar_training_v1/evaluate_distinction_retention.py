#!/usr/bin/env python3
"""Measure optional graph distinctions after corpus learning in two execution views."""
import argparse,hashlib,json,sys
from collections import Counter
from pathlib import Path
import torch
from optimizer_state import load_model
from distinction_format import tokens,ACTIONS
from retained_conditioning import evaluate

@torch.no_grad()
def run(root,tasks,checkpoints,out):
    torch.set_num_threads(1);sys.path.insert(0,str(root/'bob3ect/reentry_dnn'))
    from dtransformer_torch import DialetheicTransformerTorch
    rows=[json.loads(l) for l in (tasks/'validation.jsonl').open()];reports=[]
    for checkpoint in checkpoints:
        model=DialetheicTransformerTorch(**json.loads((checkpoint/'config.json').read_text())).double()
        model.load_state_dict(load_model(checkpoint/'model.safetensors'));model.eval()
        for view in ('flat','retained'):
            counts=Counter();error=0.
            evidence_confusion=[[0]*4 for _ in range(4)]
            for start in range(0,len(rows),64):
                selected=rows[start:start+64]
                if view=='flat':
                    sup,ref=model(torch.tensor([tokens(r) for r in selected]));st,rf=sup[:,-1,ACTIONS],ref[:,-1,ACTIONS]
                else:
                    values=[evaluate(model,tokens(r),[],context=16,kind=1,bits=0)['first_logits'] for r in selected]
                    st=torch.stack([v[0][ACTIONS] for v in values]);rf=torch.stack([v[1][ACTIONS] for v in values])
                target=torch.tensor([r['support'] for r in selected],dtype=torch.double)
                neg=torch.tensor([r['refute'] for r in selected],dtype=torch.double)
                error+=float(torch.nn.functional.binary_cross_entropy_with_logits(st,target,reduction='sum')+
                             torch.nn.functional.binary_cross_entropy_with_logits(rf,neg,reduction='sum'))
                counts['evidence_exact']+=int((((st>0)==target.bool())&((rf>0)==neg.bool())).all(-1).sum())
                for i,row in enumerate(selected):
                    for action in range(4):
                        expected=int(target[i,action])+2*int(neg[i,action])
                        proposed=int(st[i,action]>0)+2*int(rf[i,action]>0)
                        evidence_confusion[expected][proposed]+=1
                    choice=int(st[i].argmax()) if float(st[i].max())>0 else -1
                    feasible=any(row['support']);valid=choice>=0 and bool(row['support'][choice])
                    counts['cases']+=1;counts['feasible']+=feasible;counts['supported_choice']+=valid
                    counts['held_correctly']+=not feasible and choice<0
                    if choice>=0:
                        counts['revision']+=choice!=row['previous_action']
                        counts['release']+=choice==3 and row['previous_action'] in (1,2)
                        counts['supported_both_choice']+=valid and bool(row['refute'][choice])
                        counts['predicted_both_choice']+=bool(rf[i,choice]>0)
                        counts['exact_both_choice']+=valid and bool(row['refute'][choice]) and bool(rf[i,choice]>0)
            reports.append({'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256((checkpoint/'model.safetensors').read_bytes()).hexdigest(),
                            'view':view,'counts':dict(counts),'supported_choice_rate':counts['supported_choice']/counts['feasible'],
                            'evidence_exact_rate':counts['evidence_exact']/len(rows),'mean_bit_loss':error/(8*len(rows)),
                            'evidence_state_order':['Neither','Support','Refutation','Both'],
                            'evidence_confusion_target_by_proposal':evidence_confusion})
    out.write_text(json.dumps({'partition':'validation','source_sha256':hashlib.sha256((tasks/'validation.jsonl').read_bytes()).hexdigest(),
                              'evaluation':'label-free proposals compared afterward with counterfactual graph returns','reports':reports},indent=2)+'\n')
    print(json.dumps(reports))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--tasks',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,action='append',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.root.resolve(),a.tasks.resolve(),[v.resolve() for v in a.checkpoint],a.out.resolve())
