"""Exact ported-graph request and returned graph/model observation transport."""
import json,struct
from retained_format import decode_return,capsule_end,encode_state


def packet(graph,seed,vocabulary):
    ids={r['name']:r['id'] for r in vocabulary['tokens']}
    nodes=[ids[name] for name in graph['nodes']];incoming={};outgoing={};edges=[]
    for source,destination in graph['edges']:
        edges.append((source,destination,outgoing.get(source,0),incoming.get(destination,0)))
        outgoing[source]=outgoing.get(source,0)+1;incoming[destination]=incoming.get(destination,0)+1
    return b'GGRAPH1\0'+struct.pack('<4H',len(nodes),len(edges),graph['readout'],seed)+\
        struct.pack('<'+'H'*len(nodes),*nodes)+b''.join(struct.pack('<4H',*e) for e in edges)


def decode(data):
    retained_end=capsule_end(data)
    weights,retained=decode_return(data[:retained_end]);graph=data[retained_end:]
    if graph[:8]!=b'GGSTATE1':raise ValueError('missing graph state')
    n,e,sink,seed,mode,started,fields=struct.unpack_from('<7q',graph,8)
    if not 0<=n<=256 or not 0<=e<=768 or fields!=14:raise ValueError('invalid returned graph bounds')
    cursor=64;diagnostics=list(struct.unpack_from('<14q',graph,cursor));cursor+=14*8
    result={'nodes':n,'edges':e,'readout':sink,'seed':seed,'mode':mode,'started':started,'diagnostics':diagnostics}
    for key,count in [('opcodes',n),('node_inputs',n),('from',e),('to',e),('outport',e),('inport',e),
                      ('values',e),('before',e),('after',e),('expected',e)]:
        result[key]=list(struct.unpack_from('<'+'q'*count,graph,cursor));cursor+=8*count
    if len(graph)!=cursor:raise ValueError('graph return boundary mismatch')
    return weights,retained,result


def resume_packet(graph,retained):
    n,e=graph['nodes'],graph['edges']
    payload=b'GGRSTR1\0'+struct.pack('<4H',n,e,graph['readout'],graph['seed'])
    payload+=struct.pack('<'+'H'*n,*graph['opcodes'])
    payload+=b''.join(struct.pack('<4H',graph['from'][i],graph['to'][i],graph['outport'][i],graph['inport'][i]) for i in range(e))
    payload+=struct.pack('<Q14q',graph['diagnostics'][12],*graph['diagnostics'])
    payload+=struct.pack('<'+'H'*n,*graph['node_inputs'])
    payload+=b''.join(struct.pack('<4H',graph['values'][i],graph['before'][i],graph['after'][i],graph['expected'][i]) for i in range(e))
    return payload+encode_state(retained)
