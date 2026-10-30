#!/usr/bin/env python3
"""Train independent evidence over alternative IMASM graph constructions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import torch
import torch.nn.functional as F
from safetensors.torch import load_file,save_file

from train_connected import select,text_batch
from distinction_format import tokens,ACTIONS,VOCABULARY


def batch(rows):
    return (torch.tensor([tokens(r) for r in rows]),
            torch.tensor([r['support'] for r in rows],dtype=torch.float),
            torch.tensor([r['refute'] for r in rows],dtype=torch.float))


def train(root,parent,corpus,tasks,output,steps=2400):
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(29);rng=random.Random(29)
    sys.path.insert(0,str(root/'bob3ect/reentry_dnn'))
    from dtransformer_torch import DialetheicTransformerTorch
    cfg=json.loads((parent/'config.json').read_text());cfg['vocab_size']=VOCABULARY
    model=DialetheicTransformerTorch(**cfg)
    source=load_file(str(parent/'model.safetensors'))
    with torch.no_grad():
        for name,target in model.state_dict().items():
            if name in ('E_t','E_f','head.A','head.B','head.b'):
                target[:len(source[name])].copy_(source[name])
            else:target.copy_(source[name])
    rows={s:[json.loads(l) for l in (tasks/f'{s}.jsonl').open()] for s in ('train','validation','test')}
    text_pools={s:select(corpus/f'{s}.jsonl',128 if s=='train' else 24,29)
                for s in ('train','validation','test')}
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0015)

    @torch.no_grad()
    def evaluate(partition):
        model.eval();counts=Counter();traces=[];losses=[]
        for start in range(0,len(rows[partition]),128):
            selected=rows[partition][start:start+128];x,t,f=batch(selected);sup,ref=model(x)
            st,rf=sup[:,-1,ACTIONS],ref[:,-1,ACTIONS]
            losses.append(float(F.binary_cross_entropy_with_logits(st,t)+F.binary_cross_entropy_with_logits(rf,f)))
            predicted_t,predicted_f=st>0,rf>0
            counts['evidence_exact']+=int(((predicted_t==t.bool())&(predicted_f==f.bool())).all(-1).sum())
            choices=st.argmax(-1);offered=st.max(-1).values>0
            for i,row in enumerate(selected):
                choice=int(choices[i]) if offered[i] else -1
                feasible=any(row['support']);valid=choice>=0 and bool(row['support'][choice])
                counts['cases']+=1;counts['feasible']+=feasible;counts['successful_choice']+=valid
                counts['held_correctly']+=not feasible and choice<0
                if choice>=0:
                    counts[f'action_{choice}']+=1
                    counts['revision']+=choice!=row['previous_action']
                    counts['release']+=choice==3 and row['previous_action'] in (1,2)
                    counts['supported_contradictory_choice']+=valid and bool(row['refute'][choice])
                traces.append({'id':row['id'],'family':row['family'],'tokens':tokens(row),'choice':choice,
                               'support_logits':st[i].tolist(),'refute_logits':rf[i].tolist(),
                               'reference_support':row['support'],'reference_refute':row['refute'],
                               'valid_choice':valid,'previous_action':row['previous_action'],
                               'return_bits':row['candidates'][choice]['return_bits'] if choice>=0 else row['seed_bits']})
        result={'counts':dict(counts),'evidence_exact':counts['evidence_exact']/counts['cases'],
                'supported_choice_rate':counts['successful_choice']/max(1,counts['feasible']),
                'mean_evidence_loss':sum(losses)/len(losses)}
        return result,traces

    @torch.no_grad()
    def text_eval():
        result={}
        for task,samples in sorted(text_pools['test'].items()):
            x,y=text_batch(samples,64,random.Random(0));sup,_=model(x)
            result[task]=float(F.cross_entropy(sup.reshape(-1,VOCABULARY),y.reshape(-1)))
        return result

    before,_=evaluate('validation');text_before=text_eval();trace=[];started=time.monotonic()
    best_loss=float('inf');best=None;task_names=sorted(text_pools['train'])
    for step in range(steps):
        model.train();selected=rng.choices(rows['train'],k=64);x,t,f=batch(selected)
        sup,ref=model(x)
        loss=F.binary_cross_entropy_with_logits(sup[:,-1,ACTIONS],t)+F.binary_cross_entropy_with_logits(ref[:,-1,ACTIONS],f)
        if step%4==0:
            task=task_names[(step//4)%len(task_names)]
            tx,ty=text_batch(rng.choices(text_pools['train'][task],k=8),64,rng)
            ts,_=model(tx)
            loss=loss+.2*F.cross_entropy(ts.reshape(-1,VOCABULARY),ty.reshape(-1))
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if (step+1)%200==0 or step==steps-1:
            validation,_=evaluate('validation')
            row={'step':step+1,'training_loss':float(loss.detach()),'validation':validation}
            trace.append(row);print(json.dumps(row),flush=True)
            if validation['mean_evidence_loss']<best_loss:
                best_loss=validation['mean_evidence_loss'];best={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best);validation,_=evaluate('validation');test,traces=evaluate('test');text_after=text_eval()
    save_file({k:v.cpu().contiguous() for k,v in model.state_dict().items()},str(output/'model.safetensors'),
              metadata={'architecture':'dialetheic_transformer','reentry':'shared_final_trained_block','policy':'independent_candidate_evidence'})
    (output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    tokenizer={'base':'utf8_bytes_0_to_255','special':{'BOS':256,'SEP':257,'EOS':258,'carrier_TF':259,'carrier_tf':260},
               'actions':dict(zip(('direct','split_two','split_three','release'),ACTIONS)),
               'typed_fields':{'carrier_base':266,'operation_base':282,'previous_action_base':290,
                               'view_tokens':{'any':294,'fused':295,'two_arm':296,'three_arm':297},'no_negative':298},
               'sequence':['BOS','seed','work','positive_return','negative_return','previous_action','requested_view','SEP']}
    (output/'tokenizer.json').write_text(json.dumps(tokenizer,indent=2)+'\n')
    with (output/'test_choices.jsonl').open('w') as handle:
        for row in traces:handle.write(json.dumps(row)+'\n')
    probe=traces[0]
    with torch.no_grad():sup,ref=model(torch.tensor([probe['tokens']]))
    (output/'execution_input.json').write_text(json.dumps({'tokens':probe['tokens'],'source_id':probe['id'],
        'support':sup[0].tolist(),'refute':ref[0].tolist()},indent=2)+'\n')
    report={'parent':str(parent),'parent_sha256':hashlib.sha256((parent/'model.safetensors').read_bytes()).hexdigest(),
            'tasks_report_sha256':hashlib.sha256((tasks/'report.json').read_bytes()).hexdigest(),
            'steps':steps,'seed':29,'training_seconds':time.monotonic()-started,'validation_before':before,
            'validation_after':validation,'test':test,'text_loss_before':text_before,'text_loss_after':text_after,
            'training_trace':trace,'selection':'maximum learned support logit, open when all are nonpositive; refutation remains available',
            'best_checkpoint':'lowest validation evidence loss; test excluded from selection'}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'test':test,'text_loss_after':text_after}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','parent','corpus','tasks','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--steps',type=int,default=2400);a=p.parse_args()
    train(a.root.resolve(),a.parent.resolve(),a.corpus.resolve(),a.tasks.resolve(),a.out.resolve(),a.steps)
