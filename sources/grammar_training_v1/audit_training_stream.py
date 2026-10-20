#!/usr/bin/env python3
"""Read back full-record streams, their source windows, and duplicate aliases."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct


def audit(views,stream_dir):
    report=json.loads((stream_dir/'report.json').read_text())
    if report['per_task_limit'] is not None:raise ValueError('audit expects an uncapped source-record stream')
    aliases={r['example_id']:r for r in map(json.loads,(stream_dir/'duplicate_aliases.jsonl').open())}
    size=report['context_length'];seen={};counts=Counter();conditioning=Counter()
    with (stream_dir/'train.vgis').open('rb') as wire,(stream_dir/'frames.jsonl').open() as manifest:
        header=wire.read(12)
        assert header[:8]==b'VGIS1\0\0\0';declared=struct.unpack('<I',header[8:])[0]
        frame=None;frame_index=0

        def next_frame():
            nonlocal frame_index
            line=manifest.readline()
            if not line:return None
            row=json.loads(line);raw=wire.read(8+4*size)
            assert len(raw)==8+4*size
            assert hashlib.sha256(raw).hexdigest()==row['wire_sha256']
            kind,unroll,steps,bits=struct.unpack('<4H',raw[:8])
            actual_x=list(struct.unpack('<'+'H'*size,raw[8:8+2*size]))
            actual_y=list(struct.unpack('<'+'h'*size,raw[8+2*size:]))
            assert (kind,unroll,steps,bits)==tuple(row[k] for k in ('kind','unroll','steps','bits'))
            assert actual_x==row['tokens'] and actual_y==row['labels'] and row['frame']==frame_index
            frame_index+=1;counts[row['task']]+=1;return row

        frame=next_frame()
        for line in (views/'train.jsonl').open():
            source=json.loads(line)
            digest=hashlib.sha256(json.dumps([source['task'],source['prompt'],source['target']],ensure_ascii=False).encode()).hexdigest()
            if source['id'] in aliases:
                alias=aliases[source['id']]
                assert digest==seen[alias['representative']]
                assert alias['source_id']==source['source_id'] and alias['source']==source['source']
                counts['duplicate_aliases_checked']+=1;continue
            assert frame is not None and frame['kind']==0 and frame['example_id']==source['id']
            seen[source['id']]=digest
            while frame is not None and frame['kind']==0 and frame['example_id']==source['id']:
                prefix=[256]+list(source['prompt'].encode('utf8'))+[257]
                target=list(source['target'].encode('utf8'))+[258];sequence=prefix+target
                start,end=frame['input_global_span'];offset=frame['target_byte_offset'];count=frame['target_tokens']
                assert end-start==size and frame['source']==source['source'] and frame['source_id']==source['source_id']
                assert frame['family']==source['family'] and source['split']=='train'
                x=[sequence[i] if 0<=i<len(sequence) else 0 for i in range(start,end)]
                y=[sequence[i+1] if len(prefix)+offset<=i+1<len(prefix)+offset+count else -100 for i in range(start,end)]
                assert x==frame['tokens'] and y==frame['labels']
                assert sum(v>=0 for v in y)==count
                conditioning['full_prompt' if frame['full_prompt_in_context'] else 'local_context']+=1
                frame=next_frame()
        while frame is not None:
            assert frame['kind'] in (1,2)
            assert all(label== -100 for label in frame['labels'])
            frame=next_frame()
        assert frame_index==declared==report['frames'] and wire.read()==b''
    receipt={'status':'passed','frames':frame_index,'source_examples_covered':len(seen)+len(aliases),
             'counts':dict(counts),'conditioning':dict(conditioning),
             'checks':['binary/manifest equality','exact source byte windows and supervision spans',
                       'all source examples selected or linked as exact duplicates','source/family provenance retained'],
             'stream_sha256':report['bindings'][str(stream_dir/'train.vgis')]}
    (stream_dir/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--views',type=Path,required=True)
    p.add_argument('--stream',type=Path,required=True);a=p.parse_args();audit(a.views.resolve(),a.stream.resolve())
