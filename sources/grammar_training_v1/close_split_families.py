#!/usr/bin/env python3
"""Join existing source families when exact task contents bridge their splits."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import sqlite3
from pathlib import Path

from bootstrap import split_for_address
from build_training_views import Families


def rows(folder):
    for split in ('train','validation','test'):
        with (folder/f'{split}.jsonl').open() as handle:
            for line in handle:
                row=json.loads(line)
                if row['split']!=split:raise ValueError('source split mismatch')
                yield row


def fingerprint(row):
    return hashlib.sha256(json.dumps([row['prompt'],row['target']],ensure_ascii=False).encode()).hexdigest()


def close(source,output,passage_audit=None,artifact_audit=None):
    output.mkdir(parents=True,exist_ok=False)
    families=Families();first={};splits=defaultdict(set);before=Counter();count=0
    passage_rows={};passage_groups=[]
    if passage_audit:
        from audit_source_passages import normalize,SOURCE_TASKS
        audit=json.loads((passage_audit/'report.json').read_text())
        if audit['source']!=str(source):raise ValueError('passage audit belongs to another source')
        for split,digest in audit['source_sha256'].items():
            with (source/f'{split}.jsonl').open('rb') as handle:
                if hashlib.file_digest(handle,'sha256').hexdigest()!=digest:raise ValueError('passage audit source changed')
        db=sqlite3.connect((passage_audit/'passages.sqlite').as_uri()+'?mode=ro',uri=True)
        db.row_factory=sqlite3.Row
        # Include relations currently inside one split: reassigning their families
        # can otherwise create a new crossing while fixing the old crossings.
        digests=[r[0] for r in db.execute('SELECT digest FROM occurrences GROUP BY digest HAVING COUNT(DISTINCT family)>1')]
        for digest in digests:
            content=db.execute('SELECT content FROM passages WHERE digest=?',(digest,)).fetchone()[0]
            if hashlib.sha256(content.encode()).hexdigest()!=digest:raise ValueError('passage digest mismatch')
            occurrences=[dict(r) for r in db.execute('SELECT * FROM occurrences WHERE digest=?',(digest,))]
            passage_groups.append(occurrences)
            for occurrence in occurrences:
                passage_rows.setdefault(occurrence['example_id'],[]).append((occurrence,content))
        db.close()
    for row in rows(source):
        family=row['family'];families.find(family);digest=fingerprint(row)
        if digest in first:families.join(family,first[digest])
        else:first[digest]=family
        splits[digest].add(row['split']);before[row['split']]+=1;count+=1
        for span,content in passage_rows.pop(row['id'],[]):
            prefix='Continue this source passage.\n'
            if row['task'] not in SOURCE_TASKS or not row['prompt'].startswith(prefix):
                raise ValueError('passage relation needs a source continuation')
            text=row['prompt'][len(prefix):]+row['target']
            if span['family']!=family or span['split']!=row['split'] or normalize(text[span['start']:span['end']])!=content:
                raise ValueError('passage relation does not match the actual source span')
    if passage_rows:raise ValueError('passage relation has missing source examples')
    for group in passage_groups:
        first_family=group[0]['family']
        for occurrence in group[1:]:families.join(first_family,occurrence['family'])
    artifact_groups=[]
    if artifact_audit:
        audit=json.loads((artifact_audit/'report.json').read_text())
        if audit['source']!=str(source):raise ValueError('artifact audit belongs to another source')
        for path,digest in audit['input_bindings'].items():
            with Path(path).open('rb') as handle:
                if hashlib.file_digest(handle,'sha256').hexdigest()!=digest:raise ValueError('artifact audit input changed')
        bridge_path=artifact_audit/'family_bridges.jsonl'
        if hashlib.sha256(bridge_path.read_bytes()).hexdigest()!=audit['family_bridges_sha256']:
            raise ValueError('artifact family relations changed')
        known=set(families.parents)
        for line in bridge_path.open():
            group=json.loads(line);members=list(group['families'])
            if not members or any(member not in known for member in members):
                raise ValueError('artifact relation has an unknown source family')
            artifact_groups.append(members)
        for members in artifact_groups:
            for member in members[1:]:families.join(members[0],member)
    bridges=Counter('/'.join(sorted(parts)) for parts in splits.values() if len(parts)>1)
    handles={s:(output/f'{s}.jsonl').open('w',encoding='utf8') for s in ('train','validation','test')}
    counts=Counter();moved=Counter();seen_ids=set();verified={}
    try:
        for row in rows(source):
            digest=fingerprint(row);family=families.find(row['family']);split=split_for_address(family)
            if row['id'] in seen_ids:raise ValueError('duplicate source example ID')
            seen_ids.add(row['id']);moved[f'{row["split"]}->{split}']+=1
            history=list(row.get('family_history',[]))
            if 'prior_family' in row:
                previous={'family':row['prior_family'],'split':row['prior_split']}
                if not history or history[-1]!=previous:history.append(previous)
            previous={'family':row['family'],'split':row['split']}
            if not history or history[-1]!=previous:history.append(previous)
            row.update({'prior_family':row['family'],'prior_split':row['split'],
                        'family_history':history,'family':family,'split':split,
                        'dataset_revision':'artifact_family_closure_v1' if artifact_audit else 'source_passage_family_closure_v1' if passage_audit else 'exact_task_family_closure_v1'})
            handles[split].write(json.dumps(row,ensure_ascii=False)+'\n');counts[split]+=1
            if verified.setdefault(digest,split)!=split:raise RuntimeError('content still crosses splits')
    finally:
        for handle in handles.values():handle.close()
    with (output/'family_map.jsonl').open('w') as handle:
        for previous in sorted(families.parents):
            family=families.find(previous)
            handle.write(json.dumps({'prior_family':previous,'family':family,'split':split_for_address(family)})+'\n')
    sizes=Counter(families.find(f) for f in families.parents)
    inputs={}
    for name in ('train','validation','test'):
        path=source/f'{name}.jsonl'
        with path.open('rb') as handle:inputs[str(path)]=hashlib.file_digest(handle,'sha256').hexdigest()
    report={'source':str(source),'source_bindings':inputs,'records_preserved':count,
            'before_counts':dict(before),'after_counts':dict(counts),'movements':dict(moved),
            'prior_cross_split_exact_content_groups':dict(bridges),
            'remaining_cross_split_exact_content_groups':0,'prior_families':len(families.parents),
            'families':len(sizes),'largest_component_prior_families':max(sizes.values()),
            'checks':['unique example IDs','all source examples retained','exact content split separation'],
            'passage_audit':str(passage_audit) if passage_audit else None,
            'shared_passage_relations':len(passage_groups),
            'artifact_audit':str(artifact_audit) if artifact_audit else None,
            'artifact_family_components':len(artifact_groups),
            'prior_checkpoints':'not certified unseen on revised holdouts; reallocated families may have prior exposure',
            'remaining_review':['near duplicates','derivation lineage','tuple discrepancies','full conditioning for relational tasks']}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--passage-audit',type=Path,help='verified source-passage relations; constrain split families without removing records')
    p.add_argument('--artifact-audit',type=Path,help='source-bound artifact-family bridges; preserve every source example')
    a=p.parse_args();close(a.source.resolve(),a.out.resolve(),a.passage_audit.resolve() if a.passage_audit else None,
                          a.artifact_audit.resolve() if a.artifact_audit else None)
