"""Bind exact source records and provenance to a lifted SIC retrieval program."""
import hashlib,json
from pathlib import Path


def generate(source,vocabulary):
    rows=[json.loads(line) for line in source.open()];lines=[];entries=[];seen=set()
    for index,row in enumerate(rows):
        values=row['prefix']+row['target']
        if not values or len(values)>1048576 or any(type(v)is not int or not 0<=v<vocabulary for v in values):
            raise ValueError('invalid retained corpus token bank')
        if row['kind'] not in (0,1,2,3,4,5):raise ValueError('unsupported corpus operation')
        if type(row['steps'])is not int or not 0<=row['steps']<=100:raise ValueError('invalid update count')
        if row['kind'] in (3,4) and (len(row['prefix'])!=8 or row['target'] or row['bits']):raise ValueError('oracle records contain only the eight query tokens')
        if row['kind']==5 and (len(row['prefix'])!=12 or row['target'] or row['bits']):raise ValueError('checking records contain only twelve query tokens')
        key=hashlib.sha256(row['example_id'].encode()).digest()
        if key in seen:raise ValueError('duplicate exact corpus identity')
        seen.add(key)
        masses=[]
        for offset in range(0,len(values),16):
            block=values[offset:offset+16];block+= [0]*(16-len(block));total=sum(block)
            masses.extend(4*v+total for v in block)
        metadata={k:v for k,v in row.items() if k not in ('prefix','target')}
        metadata['token_bank_sha256']=hashlib.sha256(b''.join(v.to_bytes(2,'little') for v in values)).hexdigest()
        metadata['lookup_sha256']=key.hex()
        encoded=json.dumps(metadata,ensure_ascii=False,separators=(',',':')).encode()
        lines.append(f'static const unsigned short corpus_mass_{index}[]={{'+','.join(map(str,masses))+'};')
        lines.append(f'static const unsigned char corpus_meta_{index}[]={{'+','.join(map(str,encoded))+'};')
        words=[int.from_bytes(key[i:i+8],'little') for i in range(0,32,8)]
        entries.append('{'+','.join([f'corpus_mass_{index}',f'corpus_meta_{index}',str(len(encoded)),str(len(row['prefix'])),str(len(row['target'])),str(row['kind']),str(row['bits']),str(row['checkpoint_chunks']),str(row['steps'])])+',{'+','.join(str(v)+'UL' for v in words)+'}}')
    if not rows:raise ValueError('empty corpus oracle')
    lines.insert(0,'struct CorpusRecord {const unsigned short*mass;const unsigned char*metadata;unsigned long metadata_size,prefix,target,kind,bits,window,steps,key[4];};')
    lines.append('static const CorpusRecord corpus_records[]={'+','.join(entries)+'};')
    lines.append(f'static const unsigned long corpus_count={len(rows)};')
    lines.append(Path(__file__).with_name('corpus_oracle.hpp').read_text())
    return '\n'.join(lines),{'records':len(rows),'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                             'tokens':sum(len(r['prefix'])+len(r['target']) for r in rows),
                             'retrieval':'stored dimension-four Gram masses with exact signed dual synthesis',
                             'lookup':'full SHA256 of exact example ID; complete ID and provenance retained'}
