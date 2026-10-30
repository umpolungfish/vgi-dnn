"""Source-bound continuous graph memories for later executing re-entry.

The bank retains both evidence channels and probabilities independently. It
never converts an old proposal into an oracle target or drops disagreements.
"""
import copy
import hashlib
import json
import math


def binding(row):
    return hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def validate(state, row, width, clock):
    if state['graph'] != row or state['graph']['id'] != row['id']:
        raise ValueError('graph memory source identity mismatch')
    if not 0 <= state['optimizer_clock'] <= clock:
        raise ValueError('graph memory has an invalid optimizer ancestry')
    edges=len(row['graph']['edges'])
    for key,size in [('memory_t',width),('memory_f',width),('probabilities',4)]:
        values=state[key]
        if len(values)!=edges or any(len(v)!=size for v in values):
            raise ValueError('graph memory shape mismatch')
        if any(not math.isfinite(v) or not 0 <= v <= 1 for values in values for v in values):
            raise ValueError('invalid continuous graph memory')


def load(checkpoint, rows, source_sha256, vocabulary_sha256, width, clock):
    sources={r['id']:r for r in rows}
    path=checkpoint/'composed_graph_bank.json'
    if path.exists():
        bank=json.loads(path.read_text())
        if bank['source_sha256']!=source_sha256 or bank['vocabulary_sha256']!=vocabulary_sha256:
            raise ValueError('graph bank corpus or vocabulary changed')
    else:
        bank={'format':'vgi_source_bound_graph_memory_v1','source_sha256':source_sha256,
              'vocabulary_sha256':vocabulary_sha256,'states':{}}
        previous=checkpoint/'composed_graph_state.json'
        if previous.exists():
            state=json.loads(previous.read_text());identity=state['graph']['id']
            if identity not in sources:
                raise ValueError('inherited graph memory source unavailable')
            bank['states'][identity]={'binding':binding(sources[identity]),'state':state}
    for identity,entry in bank['states'].items():
        if identity not in sources or entry['binding']!=binding(sources[identity]):
            raise ValueError('graph bank identity binding mismatch')
        validate(entry['state'],sources[identity],width,clock)
    for digest,entry in bank.get('reentry_history',{}).items():
        row=entry['state']['graph'];identity=row['id']
        if identity not in sources or entry['binding']!=binding(sources[identity]) or digest!=binding(entry['state']):
            raise ValueError('graph memory history binding mismatch')
        validate(entry['state'],sources[identity],width,clock)
    return bank


def initial(bank, row, width, clock):
    entry=bank['states'].get(row['id'])
    if entry is None:return None
    if entry['binding']!=binding(row):
        raise ValueError('graph re-entry source changed')
    validate(entry['state'],row,width,clock)
    return copy.deepcopy(entry['state'])


def store(bank, state, width, clock):
    row=state['graph'];validate(state,row,width,clock)
    previous=bank['states'].get(row['id'])
    if bank.get('retain_reentry_history') and previous is not None and previous['state']!=state:
        bank.setdefault('reentry_history',{}).setdefault(binding(previous['state']),copy.deepcopy(previous))
    bank['states'][row['id']]={'binding':binding(row),'state':copy.deepcopy(state)}
