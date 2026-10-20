#!/usr/bin/env python3
"""Bind unchanged returned weights to a new corpus revision and retain prior exposure."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from retained_corpus_index import identity


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def transition(parent,source,index,out):
    metadata=json.loads(index.with_suffix('.json').read_text())
    verification=json.loads(index.with_suffix('.verification.json').read_text())
    if metadata['source']!=str(source) or metadata['source_identity']!=identity(source):
        raise ValueError('index does not bind the new source')
    if verification['status']!='passed' or verification['source_sha256']!=metadata['source_sha256'] or verification['index_sha256']!=sha(index):
        raise ValueError('new source index has no matching read-back verification')
    if sha(source)!=metadata['source_sha256']:
        raise ValueError('new source bytes changed')
    previous=json.loads((parent/'corpus_cursor.json').read_text())
    if previous['completed_records']<1:
        raise ValueError('transition requires an actually returned training boundary')
    if previous['source_sha256']==metadata['source_sha256']:
        raise ValueError('same-source continuation should retain its existing cursors')
    old_cursors=previous['next_task_cursors']
    auxiliary=previous.get('auxiliary_sources',{})
    if set(auxiliary)&set(metadata['task_counts']):
        raise ValueError('corpus task collides with a carried auxiliary task')
    cursors={task:0 for task in metadata['task_counts']}
    cursors.update({task:old_cursors.get(task,0) for task in auxiliary})
    event={'parent_checkpoint':str(parent),'parent_checkpoint_sha256':sha(parent/'model.safetensors'),
           'previous_cursor_sha256':sha(parent/'corpus_cursor.json'),'previous_cursor':previous,
           'new_source':str(source),'new_source_sha256':metadata['source_sha256'],
           'new_index':str(index),'new_index_sha256':sha(index),
           'corpus_cursors':'start a complete pass over this revision; earlier examples remain recorded as prior exposure',
           'auxiliary_cursors':'continue returned positions against unchanged source bindings',
           'holdout_exposure':'changing splits preserves all previous checkpoint exposure'}
    lineage=json.loads((parent/'dataset_lineage.json').read_text()) if (parent/'dataset_lineage.json').exists() else {'format':'vgi_dataset_lineage_v1','transitions':[]}
    lineage['transitions'].append(event)
    out.mkdir(parents=True,exist_ok=False)
    copied={}
    for name in ('model.safetensors','config.json','tokenizer.json','execution_input.json','retained_state.json'):
        path=parent/name
        if path.exists():
            shutil.copyfile(path,out/name);copied[name]=sha(path)
            assert sha(out/name)==copied[name]
    cursor={'source':str(source),'source_index':str(index),'source_sha256':metadata['source_sha256'],
            'source_task_counts':metadata['task_counts'],'next_task_cursors':cursors,
            'previous_task_cursors':cursors,'completed_records':0,'planned_records':0,
            'auxiliary_sources':auxiliary,'dataset_transition':event}
    (out/'corpus_cursor.json').write_text(json.dumps(cursor,indent=2)+'\n')
    (out/'dataset_lineage.json').write_text(json.dumps(lineage,indent=2)+'\n')
    report={'status':'passed','parent':str(parent),'out':str(out),'copied_files':copied,
            'model_and_optimizer_bytes_unchanged':True,'new_source_sha256':metadata['source_sha256'],
            'new_index_sha256':sha(index),'auxiliary_cursors':{task:cursors[task] for task in auxiliary},
            'previous_cursor_retained_exact':lineage['transitions'][-1]['previous_cursor']==previous}
    (out/'transition_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','source','index','out'):parser.add_argument('--'+name,required=True,type=Path)
    args=parser.parse_args();transition(*(getattr(args,name).resolve() for name in ('parent','source','index','out')))
