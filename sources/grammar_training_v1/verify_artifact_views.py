#!/usr/bin/env python3
"""Read back every original example and reconstruct every artifact from its views."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from build_training_views import read_rows
from close_split_families import rows


def signature(row):
    return hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def verify(source, artifacts, expanded):
    originals={row['id']:signature(row) for row in rows(source)}
    original_count=len(originals)
    inventory={row['id']:row for row in read_rows(artifacts/'artifacts.jsonl') if row['status']=='retained'}
    attestations=set();spans=defaultdict(list);seen=set();counts=Counter()
    for row in rows(expanded):
        assert row['id'] not in seen;seen.add(row['id'])
        if row['id'] in originals:
            assert signature(row)==originals.pop(row['id'])
            continue
        artifact=inventory[row['source_id']]
        assert row['source']['sha256']==artifact['sha256'] and row['source']['path']==artifact['path']
        assert row['source']['bytes']==artifact['bytes']
        if row['task']=='ob3ect_artifact_attestation':
            assert row['source_id'] not in attestations;attestations.add(row['source_id'])
            assert json.loads(row['target'])==artifact
        else:
            assert row['task']=='ob3ect_artifact_continuation'
            prefix='Continue this source passage.\n'
            assert row['prompt'].startswith(prefix)
            spans[row['source_id']].append((row['source']['start_character'],row['source']['end_character'],
                                           row['prompt'][len(prefix):]+row['target']))
        counts[row['task']]+=1
    assert not originals and attestations==set(inventory)
    text_ids={r['artifact_id'] for r in read_rows(artifacts/'text.jsonl')}
    assert set(spans)==text_ids
    for identity,parts in spans.items():
        end=0;text=[]
        for start,finish,part in sorted(parts):
            assert start==end and finish-start==len(part)
            text.append(part);end=finish
        artifact=inventory[identity];raw=''.join(text).encode('utf8')
        assert raw==(artifacts/artifact['blob']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==artifact['sha256']
    report={'status':'passed','original_examples_unchanged':original_count,'artifact_views':dict(counts),
            'complete_text_files_reconstructed':len(spans),'artifact_attestations_exact':len(attestations),
            'source':str(source),'expanded':str(expanded)}
    (expanded/'artifact_view_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','artifacts','expanded'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();verify(*(getattr(args,name).resolve() for name in ('source','artifacts','expanded')))
