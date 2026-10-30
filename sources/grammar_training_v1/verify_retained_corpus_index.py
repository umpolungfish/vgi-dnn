#!/usr/bin/env python3
"""Read back all indexed source bytes and test stale/corrupt selection rejection."""
import argparse,collections,contextlib,hashlib,io,json,sqlite3
from pathlib import Path
from retained_corpus_index import create,read


def verify(source,index,indexed_batch,linear_batch,controls):
    db=sqlite3.connect(f'file:{index}?mode=ro',uri=True)
    metadata={k:json.loads(v) for k,v in db.execute('SELECT key,value FROM metadata')}
    digest=hashlib.sha256();cursor=0;counts=collections.Counter();tokens=0
    with source.open('rb') as handle:
        for task,ordinal,offset,length,expected,row_id,family,prefix,target in db.execute('SELECT * FROM records ORDER BY offset'):
            assert offset==cursor
            line=handle.read(length);digest.update(line);assert hashlib.sha256(line).hexdigest()==expected
            row=json.loads(line);assert row['task']==task and row['id']==row_id and row['family']==family and row['split']=='train'
            assert ordinal==counts[task];counts[task]+=1
            assert prefix==len(row['prompt'].encode())+2 and target==len(row['target'].encode())+1
            cursor+=length;tokens+=prefix+target
        assert not handle.read(1)
    db.close();assert digest.hexdigest()==metadata['source_sha256'] and dict(counts)==metadata['task_counts']
    assert tokens==metadata['tokens'] and cursor==source.stat().st_size
    assert (indexed_batch/'train.vgir').read_bytes()==(linear_batch/'train.vgir').read_bytes()
    assert (indexed_batch/'records.jsonl').read_bytes()==(linear_batch/'records.jsonl').read_bytes()
    controls.mkdir(exist_ok=False)
    fragment=controls/'source.jsonl'
    with source.open('rb') as handle:fragment.write_bytes(handle.readline()+handle.readline())
    tiny=controls/'index.sqlite'
    with contextlib.redirect_stdout(io.StringIO()):create(fragment,tiny)
    read(fragment,tiny,{},2)
    corrupt=sqlite3.connect(tiny);corrupt.execute('UPDATE records SET sha256=?',('0'*64,));corrupt.commit();corrupt.close()
    try:read(fragment,tiny,{},2)
    except ValueError as e:assert str(e)=='indexed row bytes changed'
    else:raise AssertionError('corrupted row digest accepted')
    fragment.write_bytes(fragment.read_bytes()+b' ')
    try:read(fragment,tiny,{},2)
    except ValueError as e:assert 'identity changed' in str(e)
    else:raise AssertionError('changed source accepted')
    report={'status':'passed','records':sum(counts.values()),'tokens':tokens,'source_sha256':digest.hexdigest(),
            'index_sha256':hashlib.sha256(index.read_bytes()).hexdigest(),'all_row_boundaries_and_digests_match':True,
            'indexed_batch_byte_identical_to_linear_control':True,'corrupt_row_digest_rejected':True,'changed_source_rejected':True}
    index.with_suffix('.verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','index','indexed-batch','linear-batch','controls'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();verify(a.source.resolve(),a.index.resolve(),a.indexed_batch.resolve(),a.linear_batch.resolve(),a.controls.resolve())
