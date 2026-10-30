"""Share the exact common retained source state across twelve typed branches.

Only complete recurrent chunks are cached. Partial chunks stay with each axis
suffix, preserving the original chunk boundaries and absolute position labels.
The cached state stays differentiable; all axis adjoints return through it.
"""
import torch
import torch.nn.functional as F
from bootstrap import AXES,encode_address
from retained_conditioning import _chunk,evaluate


class SourcePrefix:
    def __init__(self,model,prompt):
        self.model=model
        self.common=[256]+list((prompt+'\nTyped coordinates: ').encode('utf8'))
        self.offset=len(self.common)//15*15
        self.memory=(torch.full((1,1,model.d),.05,dtype=model.E_t.dtype,device=model.E_t.device),)*2
        self.memory=self.run(self.common[:self.offset],self.memory,0)

    def run(self,tokens,memory,offset):
        model=self.model
        for start in range(0,len(tokens),15):
            count=min(15,len(tokens)-start)
            ids=torch.tensor([tokens[start:start+count]+[0]*(15-count)],device=model.E_t.device)
            positions=torch.arange(offset+start,offset+start+15,device=model.E_t.device)%model.max_len
            arguments=(model,*memory,ids,positions,count,[],torch.empty(0,dtype=torch.long,device=model.E_t.device),0,0)
            if torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                mt,mf,_,_,_=checkpoint(_chunk,*arguments,use_reentrant=False)
            else:mt,mf,_,_,_=_chunk(*arguments)
            memory=mt,mf
        return memory

    def axis(self,previous,axis):
        symbol,name,_=AXES[axis]
        suffix=list((previous+'\nAxis '+symbol+' ('+name+'): ').encode('utf8'))+[257]
        tokens=self.common[self.offset:]+suffix
        memory=self.run(tokens,self.memory,self.offset)
        return memory,len(self.common)+len(suffix)


def choices(model,prompt,previous,axis,cache=None):
    if not 0<=axis<12 or len(previous)!=axis:raise ValueError('invalid axis prefix')
    for glyph,(_,_,alphabet) in zip(previous,AXES):
        if glyph not in alphabet:raise ValueError('invalid carried coordinate')
    cache=cache or SourcePrefix(model,prompt)
    memory,count=cache.axis(previous,axis)
    scores=[]
    for glyph in AXES[axis][2]:
        branch=evaluate(model,[257],list(glyph.encode()),initial_memory=memory,differentiable_state=True)
        scores.append(-sum(branch['loss_windows']))
    return torch.stack(scores),{'memory':memory,'source_tokens_processed':count}


def objective(model,prompt,address):
    encode_address(address);cache=SourcePrefix(model,prompt);terms=[];observations=[]
    for axis,glyph in enumerate(address):
        scores,source=choices(model,prompt,address[:axis],axis,cache)
        alphabet=AXES[axis][2]
        terms.append(F.cross_entropy(scores[None],torch.tensor([alphabet.index(glyph)],device=scores.device)))
        observations.append({'axis':AXES[axis][0],'allowed':list(alphabet),'target':glyph,
            'proposal':alphabet[int(scores.detach().argmax())],'scores':scores.detach().cpu().tolist(),
            'source_tokens_processed':source['source_tokens_processed'],
            'memory_t':source['memory'][0].detach().flatten().cpu().tolist(),
            'memory_f':source['memory'][1].detach().flatten().cpu().tolist()})
    return torch.stack(terms).mean(),observations


@torch.no_grad()
def infer(model,prompt):
    cache=SourcePrefix(model,prompt);address='';observations=[]
    for axis,(symbol,_,alphabet) in enumerate(AXES):
        scores,source=choices(model,prompt,address,axis,cache)
        selected=alphabet[int(scores.argmax())];address+=selected
        observations.append({'axis':symbol,'allowed':list(alphabet),'selected':selected,
            'scores':scores.cpu().tolist(),'source_tokens_processed':source['source_tokens_processed']})
    return {'address':address,'crystal_index':encode_address(address),'observations':observations}
