"""Lossless transport of whole conditioning records and returned carrier state."""
import hashlib
import json
import struct

MAGIC=b'VGIR2\0\0\0'
STATE_FIELDS=('prefix_tokens','target_tokens','source_position','chunks','completed_records','width',
              'committed_updates','requested_updates','kind','bits','checkpoint_chunks','record_complete',
              'peak_tape_nodes','recomputed_chunks','settlement_code')


def encode_state(state):
    width=state['width'];bank=state['source_bank']
    if len(bank)!=state['prefix_tokens']+state['target_tokens']:raise ValueError('source bank length mismatch')
    result=b'VGIRM2\0\0'+struct.pack('<15Q',*(state[k] for k in STATE_FIELDS))
    for name in ('memory_t','memory_f','before_settlement_t','before_settlement_f'):
        if len(state[name])!=width:raise ValueError('memory width mismatch')
        result+=struct.pack('<'+'d'*width,*state[name])
    return result+struct.pack('<'+'H'*len(bank),*bank)


def capsule_end(data,start=0):
    if len(data)-start<8:raise IncompleteCapsule('partial checkpoint header')
    n=struct.unpack_from('<Q',data,start)[0]
    if n>len(data)-start-8:raise IncompleteCapsule('partial checkpoint metadata')
    header=json.loads(data[start+8:start+8+n])
    end=start+8+n+max(v['data_offsets'][1] for k,v in header.items() if k!='__metadata__')
    if len(data)<end+128:raise IncompleteCapsule('partial retained trailer')
    if data[end:end+8]!=b'VGIRM2\0\0':raise ValueError('invalid retained capsule marker')
    values=struct.unpack_from('<15Q',data,end+8)
    finish=end+128+32*values[5]+2*(values[0]+values[1])
    if finish>len(data):raise IncompleteCapsule('partial source bank')
    return finish


def record(row,checkpoint_chunks=4):
    prefix=[256]+list(row['prompt'].encode('utf8'))+[257]
    target=list(row['target'].encode('utf8'))+[258]
    return {'prefix':prefix,'target':target,'kind':0,'bits':0,'steps':1,'checkpoint_chunks':checkpoint_chunks,
            'example_id':row['id'],'source_id':row['source_id'],'family':row['family'],
            'task':row['task'],'source':row['source'],
            'conditioning':'all prompt tokens processed; exact prompt/target bank retained',
            'prompt_sha256':hashlib.sha256(row['prompt'].encode()).hexdigest(),
            'target_sha256':hashlib.sha256(row['target'].encode()).hexdigest()}


def pack(row):
    prefix,target=row['prefix'],row['target']
    return struct.pack('<4H2I',row['kind'],row['checkpoint_chunks'],row['steps'],row['bits'],len(prefix),len(target))+\
        struct.pack('<'+'H'*(len(prefix)+len(target)),*(prefix+target))


def stream(rows):
    return MAGIC+struct.pack('<I',len(rows))+b''.join(pack(row) for row in rows)


def decode_return(data):
    if len(data)<8:raise ValueError('missing checkpoint return')
    n=struct.unpack('<Q',data[:8])[0];header=json.loads(data[8:8+n])
    end=8+n+max(v['data_offsets'][1] for k,v in header.items() if k!='__metadata__')
    trailer=data[end:]
    if trailer[:8]!=b'VGIRM2\0\0':raise ValueError('missing retained state return')
    values=struct.unpack('<15Q',trailer[8:128])
    state=dict(zip(STATE_FIELDS,values));width=state['width'];cursor=128
    state['memory_t']=list(struct.unpack('<'+'d'*width,trailer[cursor:cursor+8*width]));cursor+=8*width
    state['memory_f']=list(struct.unpack('<'+'d'*width,trailer[cursor:cursor+8*width]));cursor+=8*width
    state['before_settlement_t']=list(struct.unpack('<'+'d'*width,trailer[cursor:cursor+8*width]));cursor+=8*width
    state['before_settlement_f']=list(struct.unpack('<'+'d'*width,trailer[cursor:cursor+8*width]));cursor+=8*width
    length=state['prefix_tokens']+state['target_tokens']
    state['source_bank']=list(struct.unpack('<'+'H'*length,trailer[cursor:cursor+2*length]));cursor+=2*length
    if len(trailer)!=cursor:raise ValueError('retained source boundary mismatch')
    state['source_bank_sha256']=hashlib.sha256(trailer[128+32*width:]).hexdigest()
    return data[:end],state


class IncompleteCapsule(ValueError):
    pass


def decode_capsules(data, allow_partial=False):
    """Read complete record snapshots, preserving a potentially interrupted tail."""
    capsules=[];offset=0
    while offset<len(data):
        start=offset
        try:
            finish=capsule_end(data,start)
            weights,state=decode_return(data[start:finish])
        except IncompleteCapsule:
            if allow_partial:return capsules,data[start:]
            raise
        capsules.append((weights,state));offset=finish
    return capsules,b''
