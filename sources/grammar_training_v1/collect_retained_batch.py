#!/usr/bin/env python3
"""Bind complete training-record returns to corpus sources and safe resume cursors."""
import argparse,hashlib,json,math,shutil
from pathlib import Path
from safetensors.torch import load_file
from retained_format import decode_capsules,stream
from optimizer_state import split_state

def collect(runtime,batch,profile,interrupted=False):
    build=json.loads((runtime/'build.json').read_text());parent=Path(build['parent'])
    status_path=profile.with_suffix('.status')
    if status_path.exists():
        status=dict(line.split('=',1) for line in status_path.read_text().splitlines())
    elif interrupted:
        status={'elapsed_seconds':'unavailable_after_interruption','samples':'unavailable_after_interruption',
                'wait_status':'interrupted_without_terminal_status','deadline_reached':'unknown',
                'binary':str((runtime/'model').resolve())}
    else:
        raise FileNotFoundError(status_path)
    assert Path(status['binary']).resolve()==(runtime/'model').resolve()
    rows=[json.loads(line) for line in (batch/'records.jsonl').open()]
    payload=(batch/'train.vgir').read_bytes();assert payload==stream(rows)
    manifest=json.loads((batch/'manifest.json').read_text())
    sha=lambda data:hashlib.sha256(data).hexdigest()
    assert sha(payload)==manifest['input_sha256']
    if build.get('corpus_entry'):
        assert sha((batch/'records.jsonl').read_bytes())==build['corpus_oracle']['source_sha256']
    capsules,partial=decode_capsules(profile.with_suffix('.stdout').read_bytes(),allow_partial=True)
    assert capsules,'no committed record return'
    oracle_traces={};record_logs={}
    for line in profile.with_suffix('.stderr').read_text().splitlines():
        if line.startswith('vgi-stream '):
            fields=list(map(int,line.split()[1:]));assert len(fields)==5
            index,kind,loss,nodes,steps=fields
            assert index not in record_logs and 0<=index<len(rows)
            assert kind==rows[index]['kind'] and loss>=0
            record_logs[index]={'loss_millionths':loss,'steps':steps,'nodes':nodes}
        if line.startswith('vgi-oracle '):
            fields=list(map(int,line.split()[1:]));assert len(fields)==14
            assert fields[0] not in oracle_traces;oracle_traces[fields[0]]=fields[1:]
    previous,parent_optimizer=split_state(load_file(str(parent/'model.safetensors')));checks=[];completed=0
    optimizer_clock=int(parent_optimizer['clock'][0]) if parent_optimizer else 0
    out=runtime/'returned_checkpoint';out.mkdir(exist_ok=True)
    for weights,state in capsules:
        count=state['completed_records'];assert 1<=count<=len(rows)
        row=rows[count-1]
        assert state['record_complete']==1
        trace=oracle_traces.get(count-1) if row['kind'] in (3,4,5) else None
        if row['kind'] in (3,4,5):
            assert trace is not None and state['kind']==row['kind'] and state['bits']==trace[0]
            if 'reference' in row:
                reference=row['reference']
                expected_bits=sum((int(reference['support'][k])<<(2*k))|(int(reference['refute'][k])<<(2*k+1)) for k in range(4))
                assert trace[0]==expected_bits
            if row['kind'] in (4,5):assert trace[0]==row['reference_bits']
            assert state['committed_updates']==trace[10] and state['requested_updates']==row['steps']
            assert trace[3]==trace[0]^trace[1] and trace[4]==trace[0]^trace[2]
            assert trace[10]==(row['steps'] if trace[3] else 0)
        else:assert state['committed_updates']==row['steps']
        assert state['source_bank']==row['prefix']+row['target']
        assert state['source_position']==len(row['prefix'])+len(row['target'])-(row['kind']==0)
        (out/'model.safetensors').write_bytes(weights)
        current,optimizer=split_state(load_file(str(out/'model.safetensors')))
        current={name:value.clone() for name,value in current.items()}
        if build.get('optimizer')=='adam':
            assert set(optimizer)=={'m','v','powers','clock'}
            assert all(bool(t.isfinite().all()) for t in optimizer.values()) and bool((optimizer['v']>=0).all())
            expected_clock=optimizer_clock+sum(record_logs[i]['steps'] for i in range(completed,count))
            assert int(optimizer['clock'][0])==expected_clock
            optimizer_clock=expected_clock
        else:assert not optimizer
        assert set(current)==set(previous)
        assert all(bool(t.isfinite().all()) for t in current.values())
        if count>completed:
            assert count>completed and all(i in record_logs for i in range(completed,count)),'missing execution receipts before snapshot'
            block_steps=sum(record_logs[i]['steps'] for i in range(completed,count))
            change=max(float((current[k].double()-previous[k].double()).abs().max()) for k in current)
            assert change>0 or block_steps==0,'no committed parameter change'
            checks.append({'record':count,'example_id':row['example_id'],'source_id':row['source_id'],
                           'task':row['task'],'tokens':len(state['source_bank']),
                           'parameter_max_change':change,'records_since_previous_snapshot':count-completed,'checkpoint_sha256':sha(weights),
                           'source_bank_exact':True,'source_position_complete':True,
                           'oracle_trace':dict(zip(('exact','before_prediction','after_prediction','before_residual','after_residual','before_choice','after_choice','before_return','after_return','after_choice_supported','correction_steps','before_loss_millionths','after_loss_millionths'),trace)) if trace is not None else None})
            previous=current;completed=count
        else:assert all(bool((current[k]==previous[k]).all()) for k in current)
        (out/'retained_state.json').write_text(json.dumps(state,indent=2)+'\n')
    record_receipts=[]
    trace_names=('exact','before_prediction','after_prediction','before_residual','after_residual','before_choice',
                 'after_choice','before_return','after_return','after_choice_supported','correction_steps',
                 'before_loss_millionths','after_loss_millionths')
    for index,row in enumerate(rows[:completed]):
        assert index in record_logs
        trace=oracle_traces.get(index) if row['kind'] in (3,4,5) else None
        if row['kind'] in (3,4,5):
            assert trace is not None
            if 'reference' in row:
                ref=row['reference'];expected_bits=sum((int(ref['support'][k])<<(2*k))|(int(ref['refute'][k])<<(2*k+1)) for k in range(4))
                assert trace[0]==expected_bits
            if row['kind'] in (4,5):assert trace[0]==row['reference_bits']
            assert trace[3]==trace[0]^trace[1] and trace[4]==trace[0]^trace[2]
            assert record_logs[index]['steps']==trace[10]==(row['steps'] if trace[3] else 0)
        else:assert record_logs[index]['steps']==row['steps']
        record_receipts.append({'record':index+1,'example_id':row['example_id'],'source_id':row['source_id'],
            'task':row['task'],'tokens':len(row['prefix'])+len(row['target']),**record_logs[index],
            'oracle_trace':dict(zip(trace_names,trace)) if trace is not None else None})
    complete=completed==len(rows) and status['wait_status']=='0' and status['deadline_reached']=='false' and not partial
    assert not complete or capsules[-1][1]['settlement_code']!=0
    cursors=dict(manifest['previous_task_cursors'])
    for row in rows[:completed]:cursors[row['task']]=row['task_ordinal']+1
    for filename in ('config.json','tokenizer.json','execution_input.json'):shutil.copyfile(parent/filename,out/filename)
    if (parent/'joint_gradient_bank.json').exists():
        shutil.copyfile(parent/'joint_gradient_bank.json',out/'joint_gradient_bank.json')
    for filename in ('composed_graph_state.json','graph_state.json','composed_anchor_observations.json','composed_source_state.json','composed_source_observations.json','composed_graph_bank.json'):
        if (parent/filename).exists():
            shutil.copyfile(parent/filename,out/filename)
    if (parent/'dataset_lineage.json').exists():
        lineage=(parent/'dataset_lineage.json').read_bytes()
        shutil.copyfile(parent/'dataset_lineage.json',out/'dataset_lineage.json')
        assert (out/'dataset_lineage.json').read_bytes()==lineage
    resume=dict(manifest,next_task_cursors=cursors,completed_records=completed,planned_records=len(rows))
    (out/'corpus_cursor.json').write_text(json.dumps(resume,indent=2)+'\n')
    report={'status':'complete' if complete else 'committed_prefix','execution':'Vox profile-native CPU execution',
            'input_binding':'baked source records; transport digest records equivalence, stdin unused' if build.get('corpus_entry') else 'VGIR2 stdin stream',
            'profile_status':status,'records':completed,'planned_records':len(rows),'checks':checks,'record_receipts':record_receipts,
            'observed_uncheckpointed_records':sorted(i+1 for i in record_logs if i>=completed),
            'tokens':sum(c['tokens'] for c in record_receipts),'partial_trailer_bytes':len(partial),
            'settlement_code':capsules[-1][1]['settlement_code'],'input_sha256':sha(payload),
            'binary_sha256':sha((runtime/'model').read_bytes()),'module_sha256':sha((runtime/'model.imasm').read_bytes()),
            'parent_checkpoint_sha256':sha((parent/'model.safetensors').read_bytes()),
            'returned_checkpoint_sha256':sha((out/'model.safetensors').read_bytes()),
            'dataset_lineage_sha256':sha((out/'dataset_lineage.json').read_bytes()) if (out/'dataset_lineage.json').exists() else None,
            'verification_scope':'exact source banks at checkpoint boundaries; execution receipts for every committed record; finite parameter changes over each checkpoint interval; independent gradient checks have their own receipts'}
    (runtime/'corpus_training_receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','records','tokens','settlement_code')}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('runtime','batch','profile'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--interrupted',action='store_true',help='collect only complete saved capsules when profiler status is absent')
    a=p.parse_args();collect(a.runtime.resolve(),a.batch.resolve(),a.profile.resolve(),a.interrupted)
