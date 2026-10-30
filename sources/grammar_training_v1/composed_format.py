"""Decode the model, discrete graph, and continuous composed carrier return."""
import math
import struct
from retained_format import capsule_end
from graph_format import decode as decode_graph


def decode(data):
    retained_end = capsule_end(data)
    n, e = struct.unpack_from('<2q', data, retained_end + 8)
    graph_end = retained_end + 176 + 16*n + 64*e
    weights, retained, graph = decode_graph(data[:graph_end]); tail = data[graph_end:]
    if tail[:8] != b'VGICMP1\0':raise ValueError('missing composed state')
    values = struct.unpack_from('<11q', tail, 8)
    keys = ('version','nodes','edges','width','seed','source_return','passes','updates','detach','started','queries')
    state = dict(zip(keys, values))
    if state['version'] not in (1,2,3,4,5) or state['nodes'] != n or state['edges'] != e or state['width'] != retained['width']:
        raise ValueError('composed state dimensions differ')
    if not 0 <= e <= 128 or not 0 <= n <= 256:raise ValueError('invalid composed state bounds')
    state['binding'] = struct.unpack_from('<Q', tail, 96)[0]
    state['loss'],state['local_loss'],state['terminal_loss'] = struct.unpack_from('<3d',tail,104)
    cursor = 128
    for name,count in (('source_nodes',n),('source_edges',e),('local_targets',e)):
        state[name] = list(struct.unpack_from('<'+'q'*count,tail,cursor));cursor += 8*count
    state['probabilities'] = [];state['memory_t'] = [];state['memory_f'] = []
    width = state['width']
    for _ in range(e):
        for name,count in (('probabilities',4),('memory_t',width),('memory_f',width)):
            state[name].append(list(struct.unpack_from('<'+'d'*count,tail,cursor)));cursor += 8*count
    if state['version'] >= 2:
        state['memory_gradients'] = []
        for _ in range(e):
            state['memory_gradients'].append(list(struct.unpack_from('<'+'d'*(2*width),tail,cursor)));cursor += 16*width
    if state['version'] >= 3:
        state['trace_model'],count=struct.unpack_from('<Qq',tail,cursor);cursor+=16
        if not 0<=count<=2048:raise ValueError('invalid composed event count')
        state['trace']=[]
        for _ in range(count):
            event=dict(zip(('pass','edge','input','target'),struct.unpack_from('<4q',tail,cursor)));cursor+=32
            for name,size in (('memory',2*width),('memory_gradient',2*width),('probability',4),('probability_gradient',4)):
                event[name]=list(struct.unpack_from('<'+'d'*size,tail,cursor));cursor+=8*size
            state['trace'].append(event)
    if state['version']>=5:
        state['source_diagnostics']=list(struct.unpack_from('<14q',tail,cursor));cursor+=112
    if state['version']>=4:
        state['oracle_calls'],state['model_fingerprint'],state['retained_fingerprint'],state['checksum']=struct.unpack_from('<4Q',tail,cursor);cursor+=32
    if len(tail) != cursor:raise ValueError('composed return boundary differs')
    for name in ('probabilities','memory_t','memory_f'):
        if any(not math.isfinite(v) or not 0 <= v <= 1 for row in state[name] for v in row):
            raise ValueError('invalid continuous carried evidence')
    return weights,retained,graph,state


def restore_packet(returned, passes=0,steps=0,local_weight=0.,detach=False):
    """Restore the complete returned carrier into its matching compiled model."""
    from retained_format import encode_state
    weights,retained,graph,state=decode(returned)
    if state['version']!=5:raise ValueError('restoration requires a bound composed version-five return')
    retained_end=capsule_end(returned);graph_end=retained_end+176+16*graph['nodes']+64*graph['edges']
    if not 0<=passes<=16 or not 0<=steps<=100 or not 0<=local_weight<=100 or (not passes and steps):raise ValueError('invalid composed continuation request')
    return b'CMREENT1'+struct.pack('<4Q',passes,steps,round(local_weight*1000000),int(detach))+returned[retained_end:graph_end]+returned[graph_end:]+encode_state(retained)
