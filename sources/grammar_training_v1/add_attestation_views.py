#!/usr/bin/env python3
"""Keep catalog and ob3ect attestations trainable while typed views await review."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from bootstrap import split_for_address


def add(root,source,linked,original_families,output):
    output.mkdir(parents=True,exist_ok=False)
    old={r['source_id']:r['family'] for r in map(json.loads,original_families.open())}
    revised={r['prior_family']:r['family'] for r in map(json.loads,(source/'family_map.jsonl').open())}
    counts=Counter();issues=[];maximum=0
    handles={s:(output/f'{s}.jsonl').open('w',encoding='utf8') for s in ('train','validation','test')}
    try:
        for split in handles:
            with (source/f'{split}.jsonl').open() as handle:
                for line in handle:
                    row=json.loads(line);handles[split].write(line);counts[f'{split}:{row["task"]}']+=1

        def emit(rid,kind,name,payload,record_source,family,codec_status=None):
            nonlocal maximum
            task=kind+'_attestation';split=split_for_address(family)
            prompt=('Reproduce the saved '+kind+' attestation for '+str(name)+'.\n'+
                    'Source SHA256: '+record_source['sha256']+'\nFields describe the saved record.')
            target=json.dumps(payload,ensure_ascii=False) if not isinstance(payload,str) else payload
            row={'id':rid+':'+task,'task':task,'source_id':rid,'family':family,'split':split,
                 'prompt':prompt,'target':target,'source':record_source,'evidence':'source_attestation',
                 'tuple_codec_status':codec_status,'live_execution':None,
                 'dataset_revision':'whole_record_attestations_v1'}
            handles[split].write(json.dumps(row,ensure_ascii=False)+'\n');counts[f'{split}:{task}']+=1
            maximum=max(maximum,len(prompt.encode())+len(target.encode())+3)

        for line in (linked/'records.jsonl').open():
            record=json.loads(line)
            if record['kind'] not in ('catalog_entry','ob3ect'):continue
            rid=record['id'];family=old[rid];family=revised.get(family,family)
            payload=record['payload'];kind='catalog' if record['kind']=='catalog_entry' else 'ob3ect'
            emit(rid,kind,payload.get('name'),payload,record['source'],family,
                 'passed_existing_codec' if 'tuple' in record else 'pending_or_unavailable')

        for error in json.loads((linked/'errors.json').read_text()):
            source_record=error.get('source')
            if not source_record or source_record.get('kind')!='ob3ect':continue
            path=root/source_record['path']
            try:
                raw=path.read_bytes()
                if hashlib.sha256(raw).hexdigest()!=source_record['sha256']:
                    raise ValueError('source bytes changed since inventory')
                text=raw.decode('utf8')
            except (OSError,UnicodeError,ValueError) as exc:
                issues.append({'source':source_record,'error':str(exc)});continue
            rid='raw-ob3ect:'+source_record['sha256']+':'+source_record['path']
            emit(rid,'ob3ect_raw',path.stem,text,{**source_record,'extraction_error':error['error']},
                 'raw-ob3ect:'+source_record['sha256'])
    finally:
        for handle in handles.values():handle.close()
    bindings={}
    for path in (source/'report.json',linked/'records.jsonl',linked/'errors.json',original_families):
        with path.open('rb') as handle:bindings[str(path)]=hashlib.file_digest(handle,'sha256').hexdigest()
    report={'task_counts':dict(counts),'maximum_attestation_tokens_utf8':maximum,
            'records_preserved':'all previous examples copied; whole-record attestations added',
            'source_bindings':bindings,'issues':issues,
            'interpretation':'saved flags and tuples remain source observations; no new proof or execution status inferred',
            'next_check':'close exact task-family overlaps on this expanded dataset before evaluation'}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','source','linked','original-families','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();add(a.root.resolve(),a.source.resolve(),a.linked.resolve(),a.original_families.resolve(),a.out.resolve())
