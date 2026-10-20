#!/usr/bin/env python3
"""Find shared normalized source passages and retain their split/provenance links."""
import argparse,hashlib,json,re,sqlite3,unicodedata
from collections import Counter
from pathlib import Path

SOURCE_TASKS={'document_continuation','lean_source_continuation','ob3ect_artifact_continuation'}

def normalize(text):
    return ' '.join(unicodedata.normalize('NFC',text).split())


def audit(source,out):
    out.mkdir(parents=True,exist_ok=False)
    db=sqlite3.connect(out/'passages.sqlite')
    db.executescript('''CREATE TABLE passages(digest TEXT PRIMARY KEY,content TEXT NOT NULL);
CREATE TABLE occurrences(digest TEXT,example_id TEXT,family TEXT,split TEXT,source_id TEXT,
 source_path TEXT,source_sha256 TEXT,source_start_line INTEGER,passage_kind TEXT,start INTEGER,end INTEGER);
CREATE INDEX occurrence_digest ON occurrences(digest);''')
    bindings={};counts=Counter()
    for split in ('train','validation','test'):
        path=source/f'{split}.jsonl'
        with path.open('rb') as handle:bindings[split]=hashlib.file_digest(handle,'sha256').hexdigest()
        with path.open() as handle:
            for line in handle:
                row=json.loads(line)
                if row['task'] not in SOURCE_TASKS:continue
                prefix='Continue this source passage.\n'
                assert row['prompt'].startswith(prefix) and row['split']==split
                text=row['prompt'][len(prefix):]+row['target'];counts['source_examples']+=1
                spans=[('whole_chunk',0,len(text))]
                start=0
                for boundary in re.finditer(r'\n[ \t]*\n',text):
                    spans.append(('paragraph',start,boundary.start()));start=boundary.end()
                spans.append(('paragraph',start,len(text)));seen=set()
                for kind,start,end in spans:
                    content=normalize(text[start:end])
                    # Keep literal source text; only whitespace and NFC normalization unite spans.
                    if len(content)<256 or len(set(re.findall(r'\w+',content)))<32:continue
                    digest=hashlib.sha256(content.encode()).hexdigest()
                    if digest in seen:continue
                    seen.add(digest);counts['occurrences']+=1
                    previous=db.execute('SELECT content FROM passages WHERE digest=?',(digest,)).fetchone()
                    if previous:assert previous[0]==content
                    else:db.execute('INSERT INTO passages VALUES (?,?)',(digest,content))
                    provenance=row['source']
                    db.execute('INSERT INTO occurrences VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                        (digest,row['id'],row['family'],split,row['source_id'],provenance.get('path'),
                         provenance.get('sha256'),provenance.get('start_line'),kind,start,end))
        db.commit()
    groups=list(db.execute('''SELECT digest,COUNT(*),COUNT(DISTINCT family),COUNT(DISTINCT split)
 FROM occurrences GROUP BY digest HAVING COUNT(DISTINCT family)>1 ORDER BY digest'''))
    cross=[g for g in groups if g[3]>1];counts['shared_passage_groups']=len(groups);counts['cross_split_groups']=len(cross)
    with (out/'cross_split_passages.jsonl').open('w') as handle:
        for digest,total,families,splits in cross:
            content=db.execute('SELECT content FROM passages WHERE digest=?',(digest,)).fetchone()[0]
            columns=[c[1] for c in db.execute('PRAGMA table_info(occurrences)')]
            occurrences=[dict(zip(columns,r)) for r in db.execute('SELECT * FROM occurrences WHERE digest=? ORDER BY example_id',(digest,))]
            handle.write(json.dumps({'digest':digest,'normalized_passage':content,'occurrences':occurrences},ensure_ascii=False)+'\n')
    counts['unique_passages']=db.execute('SELECT COUNT(*) FROM passages').fetchone()[0]
    db.close()
    # Re-read each implicated example from the immutable training snapshots and
    # verify every recorded span against its normalized retained passage.
    wanted={}
    for line in (out/'cross_split_passages.jsonl').open():
        group=json.loads(line)
        for occurrence in group['occurrences']:
            wanted.setdefault(occurrence['example_id'],[]).append((occurrence,group['normalized_passage']))
    checked=0
    for split in bindings:
        path=source/f'{split}.jsonl'
        with path.open() as handle:
            for line in handle:
                row=json.loads(line)
                if row['id'] not in wanted:continue
                text=row['prompt'][len('Continue this source passage.\n'):]+row['target']
                for span,expected in wanted.pop(row['id']):
                    assert normalize(text[span['start']:span['end']])==expected
                    assert span['family']==row['family'] and span['split']==split
                    checked+=1
        with path.open('rb') as handle:assert hashlib.file_digest(handle,'sha256').hexdigest()==bindings[split]
    assert not wanted
    report={'source':str(source),'source_sha256':bindings,'counts':dict(counts),'verified_cross_split_occurrences':checked,
            'normalization':'Unicode NFC and whitespace collapse; case, names, punctuation, and evidence atoms preserved',
            'minimum_characters':256,'minimum_unique_words':32,
            'relation':'shared literal normalized passage; retains each construction and its source identity',
            'next_use':'passage links constrain future split families; existing checkpoint exposure remains attached to its original data'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();audit(a.source.resolve(),a.out.resolve())
