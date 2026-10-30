"""Independent shared-transformer computation over the retained graph edges."""
import torch
import torch.nn.functional as F
from retained_conditioning import evaluate


def forward(model, row, vocabulary, oracle, passes, local_weight=.25, detach=False, initial=None):
    from dtransformer_torch import logit
    graph = row['graph']; seed = row['seed_bits']; edges = graph['edges']; width = model.d
    ids = {r['name']:r['id'] for r in vocabulary['tokens']}
    device = next(model.parameters()).device
    memory = [(torch.full((1,1,width),.05,dtype=torch.double,device=device),torch.full((1,1,width),.05,dtype=torch.double,device=device)) for _ in edges]
    probability = [torch.zeros(4,dtype=torch.double,device=device) for _ in edges]
    if initial:
        memory = [(torch.tensor(initial['memory_t'][i],dtype=torch.double,device=device).reshape(1,1,width),
                   torch.tensor(initial['memory_f'][i],dtype=torch.double,device=device).reshape(1,1,width)) for i in range(len(edges))]
        probability = [torch.tensor(v,dtype=torch.double,device=device) for v in initial['probabilities']]
    fans = {i:sum(a==i for a,b in edges) for i in range(len(graph['nodes']))}
    local = []; node_inputs = [0]*len(graph['nodes']); targets = [0]*len(edges); terminal = None;trace=[]
    readout_target = torch.tensor([(row['return_bits']>>bit)&1 for bit in range(4)],dtype=torch.double,device=device)
    for pass_index in range(passes):
        for node,name in enumerate(graph['nodes']):
            incoming = [i for i,(a,b) in enumerate(edges) if b==node]
            if incoming:
                mt,mf = memory[incoming[0]];p = probability[incoming[0]]
                for i in incoming[1:]:
                    mt,mf = torch.maximum(mt,memory[i][0]),torch.maximum(mf,memory[i][1]);p = torch.maximum(p,probability[i])
            else:
                mt=mf=torch.full((1,1,width),.05,dtype=torch.double,device=device);p=torch.zeros(4,dtype=torch.double,device=device)
            bits = sum(int(v>.5)<<k for k,v in enumerate(p.detach().tolist()));node_inputs[node] = bits
            if node==graph['readout']:
                clipped = p.clamp(1e-6,1-1e-6);z = torch.log(clipped/(1-clipped))
                terminal = F.binary_cross_entropy_with_logits(z,readout_target)
            outgoing = [i for i,(a,b) in enumerate(edges) if a==node]
            for slot,edge in enumerate(outgoing):
                expected = oracle(ids[name],bits,seed,slot,fans[node]);assert 0 <= expected < 16
                query = [256,71,ids[name],266+bits,266+seed,fans[node],slot,257]
                carried = (mt.detach(),mf.detach()) if detach else (mt,mf)
                result = evaluate(model,query,[],context=9,kind=2,bits=expected,
                                  initial_memory=carried,differentiable_state=True)
                memory[edge] = result['memory']
                # The edge state is also the input of its own carrier readout,
                # matching the native Var identities and their local pullbacks.
                nt,nf=model.final_norm(*memory[edge]);a,b=model.head(logit(nt),logit(nf));a,b=a[0,0],b[0,0]
                z=torch.stack((a[259],b[259],a[260],b[260]));probability[edge]=torch.sigmoid(z)
                target=torch.tensor([(expected>>bit)&1 for bit in range(4)],dtype=torch.double,device=device)
                local.append(F.binary_cross_entropy_with_logits(z,target));targets[edge] = expected
                trace.append({'pass':pass_index,'edge':edge,'input':bits,'target':expected,
                              'memory':memory[edge],'probability':probability[edge]})
    local_mean = sum(local)/len(local) if local else torch.tensor(0.,dtype=torch.double,device=device)
    return {'loss':terminal+local_weight*local_mean,'local_loss':local_mean,'terminal_loss':terminal,
            'probabilities':probability,'memory':memory,'node_inputs':node_inputs,'local_targets':targets,
            'returned':node_inputs[graph['readout']],'trace':trace}
