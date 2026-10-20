#!/usr/bin/env python3
"""Index immutable corpus row boundaries for exact, resumable batch selection."""
import argparse,collections,hashlib,json,sqlite3
from pathlib import Path


def identity(path):
    stat=path.stat()
    return {name:getattr(stat,name) for name in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')}


def create(source,output):
    if output.exists():raise FileExistsError(output)
    before=identity(source);output.parent.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(output);db.executescript('''
        CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE records(task TEXT NOT NULL,ordinal INTEGER NOT NULL,offset INTEGER NOT NULL,
            length INTEGER NOT NULL,sha256 TEXT NOT NULL,id TEXT NOT NULL,family TEXT NOT NULL,
            prefix_tokens INTEGER NOT NULL,target_tokens INTEGER NOT NULL,PRIMARY KEY(task,ordinal));
    ''')
    counts=collections.Counter();digest=hashlib.sha256();maximum=tokens=0
    try:
        with source.open('rb') as handle:
            while True:
                offset=handle.tell();line=handle.readline()
                if not line:break
                digest.update(line);row=json.loads(line)
                if row['split']!='train':raise ValueError('non-training row in corpus index')
                prefix=len(row['prompt'].encode('utf8'))+2;target=len(row['target'].encode('utf8'))+1
                if prefix+target>1048576:raise ValueError('complete source exceeds executing bank: '+row['id'])
                task=row['task'];ordinal=counts[task];counts[task]+=1
                db.execute('INSERT INTO records VALUES(?,?,?,?,?,?,?,?,?)',
                    (task,ordinal,offset,len(line),hashlib.sha256(line).hexdigest(),row['id'],row['family'],prefix,target))
                tokens+=prefix+target;maximum=max(maximum,prefix+target)
        if identity(source)!=before:raise ValueError('source changed while indexing')
        metadata={'format':'retained_corpus_index_v1','source':str(source),'source_identity':before,
                  'source_sha256':digest.hexdigest(),'task_counts':dict(counts),'records':sum(counts.values()),
                  'tokens':tokens,'max_record_tokens':maximum}
        db.executemany('INSERT INTO metadata VALUES(?,?)',[(k,json.dumps(v)) for k,v in metadata.items()]);db.commit()
        assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    finally:db.close()
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps(metadata))


def read(source,index,cursors,limit):
    db=sqlite3.connect(f'file:{index}?mode=ro',uri=True)
    try:
        metadata={k:json.loads(v) for k,v in db.execute('SELECT key,value FROM metadata')}
        if metadata['source']!=str(source) or metadata['source_identity']!=identity(source):
            raise ValueError('indexed source identity changed; retain the old index and rebuild against the intended corpus')
        queues=collections.defaultdict(list)
        with source.open('rb') as handle:
            for task in sorted(metadata['task_counts']):
                for ordinal,offset,length,digest,row_id,prefix,target in db.execute(
                        'SELECT ordinal,offset,length,sha256,id,prefix_tokens,target_tokens FROM records '
                        'WHERE task=? AND ordinal>=? ORDER BY ordinal LIMIT ?', (task,cursors.get(task,0),limit)):
                    handle.seek(offset);line=handle.read(length)
                    if hashlib.sha256(line).hexdigest()!=digest:raise ValueError('indexed row bytes changed')
                    row=json.loads(line)
                    if row['id']!=row_id or row['task']!=task or row['split']!='train':raise ValueError('indexed row identity changed')
                    if len(row['prompt'].encode())+2!=prefix or len(row['target'].encode())+1!=target:raise ValueError('indexed token count changed')
                    queues[task].append((ordinal,row))
        if metadata['source_identity']!=identity(source):raise ValueError('indexed source changed during selection')
        return queues,collections.Counter(metadata['task_counts']),metadata['source_sha256']
    finally:db.close()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();create(a.source.resolve(),a.out.resolve())
