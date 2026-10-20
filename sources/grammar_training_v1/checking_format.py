"""Checking requests and exact returned continuation capsules."""
import struct
from retained_format import capsule_end,decode_return,encode_state,STATE_FIELDS

FIELDS=('active','phase','sequence','oracle_calls','model_fingerprint','query_length',
        'trace_fields','teacher_fields','retained_fingerprint','state_checksum')
MASK=(1<<64)-1

def mix(values):
    result=14695981039346656037
    for value in values:result=((result^(value&MASK))*1099511628211)&MASK
    return result

def retained_fingerprint(state):
    values=[state[k] for k in STATE_FIELDS]
    for name in ('memory_t','memory_f','before_settlement_t','before_settlement_f'):
        values.extend(struct.unpack('<Q',struct.pack('<d',value))[0] for value in state[name])
    return mix(values+state['source_bank'])

def encode_checking(state):
    header=[state[k] for k in FIELDS]
    context=[struct.unpack('<Q',struct.pack('<d',value))[0] for name in ('initial_t','initial_f') for value in state[name]]
    checksum=mix(header[:9]+state['query']+state['trace']+state['teacher_trace']+context)
    if checksum!=header[9]:raise ValueError('checking state checksum mismatch')
    return b'CGSTATE2'+struct.pack('<10Q',*header)+struct.pack('<12H',*state['query'])+\
        struct.pack('<8q',*state['trace'])+struct.pack('<13q',*state['teacher_trace'])+\
        b''.join(struct.pack('<'+'d'*len(state[name]),*state[name]) for name in ('initial_t','initial_f'))

def decode(data):
    end=capsule_end(data);weights,retained=decode_return(data[:end]);payload=data[end:]
    if len(payload)!=280+16*retained['width'] or payload[:8]!=b'CGSTATE2':raise ValueError('checking return boundary mismatch')
    state=dict(zip(FIELDS,struct.unpack_from('<10Q',payload,8)))
    state['query']=list(struct.unpack_from('<12H',payload,88))
    state['trace']=list(struct.unpack_from('<8q',payload,112))
    state['teacher_trace']=list(struct.unpack_from('<13q',payload,176))
    for i,name in enumerate(('initial_t','initial_f')):
        state[name]=list(struct.unpack_from('<'+'d'*retained['width'],payload,280+8*i*retained['width']))
    if encode_checking(state)!=payload:raise ValueError('checking state reconstruction mismatch')
    if state['retained_fingerprint']!=retained_fingerprint(retained):raise ValueError('retained fingerprint mismatch')
    return weights,retained,state

def packet(query,rounds=0,steps=0,readout=3):
    if len(query)!=12:raise ValueError('twelve checking tokens required')
    return b'CGRUN1\0\0'+struct.pack('<4Q',0,rounds,steps,readout)+struct.pack('<12H',*query)

def resume_packet(state,retained,rounds=0,steps=0,readout=3):
    return b'CGRUN1\0\0'+struct.pack('<4Q',2 if rounds else 1,rounds,steps,readout)+encode_checking(state)+encode_state(retained)
