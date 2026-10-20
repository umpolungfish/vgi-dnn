#!/usr/bin/env python3
"""Continue source-indexed, oracle-integrated IMASM learning with durable receipts."""
import argparse,contextlib,fcntl,hashlib,json,os,signal,subprocess,time
from pathlib import Path
from prepare_retained_batch import build as prepare
from add_distinction_replay import build as replay
from export_full_learning import export
from collect_retained_batch import collect
from add_graph_replay import build as graph_replay
from add_operation_replay import build as operation_replay
from train_graph_trajectories import train as train_trajectories
from export_joint_retention import build as build_joint
from verify_joint_retention import verify as verify_joint
from collect_joint_retention import run as collect_joint

IMPLEMENTATION=('train_corpus_campaign.py','prepare_retained_batch.py','retained_corpus_index.py',
    'add_distinction_replay.py','export_full_learning.py','export_imasm_model.py','corpus_oracle.py',
    'corpus_oracle.hpp','retained_learning.hpp','retained_oracle.hpp','reverse_tape.hpp',
    'stream_learning.hpp','dialetheic_runtime.c','retained_format.py','distinction_format.py','collect_retained_batch.py',
    'graph_learning.hpp','add_graph_replay.py','optimizer_adam.hpp','optimizer_state.py',
    'checking_policy.hpp','add_operation_replay.py','retained_io.hpp','checking_return.hpp',
    'train_graph_trajectories.py','export_graph_learning.py','graph_format.py',
    'export_joint_retention.py','verify_joint_retention.py','collect_joint_retention.py',
    'train_cuda_trajectories.py','composed_graph_reference.py','retained_conditioning.py',
    'export_composed_graph.py','composed_graph.hpp','composed_io.hpp',
    'typed_tuple_learning.py','bootstrap.py','graph_reentry_bank.py',
    'train_banked_cuda_trajectories.py','cached_typed_tuple_learning.py','distinction_rehearsal.py','paired_language_learning.py')

def hashes():
    here=Path(__file__).parent
    return {name:hashlib.sha256((here/name).read_bytes()).hexdigest() for name in IMPLEMENTATION}

def save(path,value):
    pending=path.with_suffix(path.suffix+'.pending');pending.write_text(json.dumps(value,indent=2)+'\n');os.replace(pending,path)

def run(a):
    if a.composed_graph_weight != 1 and not a.graph_reentry:
        raise ValueError('custom graph objective weight requires the banked CUDA stage')
    if a.composed_distinction_weight and (not a.graph_reentry or a.trajectory_backend != 'cuda-composed'):
        raise ValueError('paired distinction rehearsal requires the banked CUDA stage')
    a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        initial=hashes();state_path=a.out/'campaign.json'
        graph_bindings={name:{'path':str(getattr(a,name)),'sha256':hashlib.sha256(getattr(a,name).read_bytes()).hexdigest()} for name in ('graph_records','graph_library','graph_vocabulary')} if a.graph_records else None
        checking_binding={'path':str(a.checking_tasks),'sha256':hashlib.sha256(a.checking_tasks.read_bytes()).hexdigest()} if a.checking_tasks else None
        configuration={'token_budget':a.token_budget,'max_records':a.max_records,'oracle_queries':a.oracle_queries,
                       'snapshot_records':a.snapshot_records,'learning_rate':a.learning_rate,'duration_ms':a.duration_ms,
                       'graph_queries':a.graph_queries,'graph_bindings':graph_bindings,'optimizer':a.optimizer,
                       'checking_queries':a.checking_queries,'checking_binding':checking_binding}
        trajectory_bindings=None
        if a.trajectory_dataset:
            trajectory_bindings={name:{'path':str(a.trajectory_dataset/name),
                'sha256':hashlib.sha256((a.trajectory_dataset/name).read_bytes()).hexdigest()}
                for name in ('transitions.jsonl','vocabulary.json')}
            configuration['trajectories']={'bindings':trajectory_bindings,'count':a.trajectory_count,
                'mode':a.trajectory_mode,'steps':a.trajectory_steps,'seed':a.trajectory_seed}
            if a.trajectory_backend=='cuda-composed':
                configuration['trajectories'].update(backend=a.trajectory_backend,devices=a.cuda_devices,
                    passes=a.composed_passes,local_weight=a.composed_local_weight,
                    anchor_weight=a.composed_anchor_weight,anchor_count=a.composed_anchor_count,
                    source_weight=a.composed_source_weight,typed_tuples=a.typed_tuples,
                    graph_reentry=a.graph_reentry,graph_weight=a.composed_graph_weight,distinction_weight=a.composed_distinction_weight,
                    distinction_count=a.composed_distinction_count)
        joint_binding=None
        if a.joint_graph_source:
            joint_binding={'path':str(a.joint_graph_source),'sha256':hashlib.sha256(a.joint_graph_source.read_bytes()).hexdigest()}
            configuration['joint_gradient']={'source':joint_binding,'weight':a.joint_weight,
                'anchors':a.joint_anchors,'pool':a.joint_pool,'balanced_evidence':a.joint_balanced_evidence}
        if state_path.exists():
            state=json.loads(state_path.read_text())
            if state['implementation']!=initial:raise ValueError('campaign implementation changed; retain this run and start a new campaign from its returned checkpoint')
            if state['configuration']!=configuration or state['source']!=str(a.source) or state['index']!=str(a.index) or state['tasks']!=str(a.tasks):
                raise ValueError('campaign configuration changed; retain this run and start a new campaign from its returned checkpoint')
            checkpoint=Path(state['checkpoint']);cursor=Path(state['cursor']);next_batch=state['next_batch']
            if state.get('inflight'):raise ValueError('campaign has an interrupted in-flight batch; collect its existing profiler return before starting more work')
        else:
            checkpoint=a.checkpoint;cursor=a.cursor;next_batch=0
            state={'status':'prepared','checkpoint':str(checkpoint),'cursor':str(cursor),'next_batch':0,
                   'implementation':initial,'completed_batches':[],'source':str(a.source),'index':str(a.index),
                   'tasks':str(a.tasks),'configuration':configuration}
        stopping=False
        def stop(signum,frame):
            nonlocal stopping
            stopping=True
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        launched=0
        while a.batches==0 or launched<a.batches:
            if stopping or (a.out/'STOP').exists():state['status']='stopped_at_boundary';save(state_path,state);return
            if graph_bindings and any(hashlib.sha256(Path(v['path']).read_bytes()).hexdigest()!=v['sha256'] for v in graph_bindings.values()):
                state['status']='graph_binding_changed';save(state_path,state);return
            if checking_binding and hashlib.sha256(a.checking_tasks.read_bytes()).hexdigest()!=checking_binding['sha256']:
                state['status']='checking_binding_changed';save(state_path,state);return
            if trajectory_bindings and any(hashlib.sha256(Path(v['path']).read_bytes()).hexdigest()!=v['sha256'] for v in trajectory_bindings.values()):
                state['status']='trajectory_binding_changed';save(state_path,state);return
            if joint_binding and hashlib.sha256(a.joint_graph_source.read_bytes()).hexdigest()!=joint_binding['sha256']:
                state['status']='joint_binding_changed';save(state_path,state);return
            if hashes()!=initial:state['status']='implementation_changed';save(state_path,state);return
            if (checkpoint/'corpus_cursor.json').exists() and json.loads(cursor.read_text())!=json.loads((checkpoint/'corpus_cursor.json').read_text()):
                raise ValueError('checkpoint and corpus cursor do not share a returned boundary')
            batch_root=a.out/f'batch_{next_batch:06d}';batch_root.mkdir(exist_ok=False)
            state.update(status='preparing',inflight=str(batch_root));save(state_path,state)
            with (batch_root/'preparation.log').open('w') as log,contextlib.redirect_stdout(log):
                prepare(a.source,batch_root/'corpus',a.token_budget,a.max_records,cursor,a.index)
                manifest=json.loads((batch_root/'corpus/manifest.json').read_text())
                if manifest['records']==0:
                    state.update(status='corpus_epoch_complete',inflight=None);save(state_path,state);return
                replay(batch_root/'corpus',a.tasks,batch_root/'distinctions' if a.graph_records or a.checking_tasks else batch_root/'input',a.oracle_queries,cursor,True)
                if a.graph_records:graph_replay(batch_root/'distinctions',a.graph_records,batch_root/'ports' if a.checking_tasks else batch_root/'input',a.graph_queries,cursor)
                if a.checking_tasks:operation_replay(batch_root/('ports' if a.graph_records else 'distinctions'),a.checking_tasks,batch_root/'input',a.checking_queries,cursor,'checking_policy',5,43,True)
                if joint_binding:
                    build_joint(a.root,checkpoint,batch_root/'input/records.jsonl',a.joint_graph_source,
                        a.graph_library,a.graph_vocabulary,batch_root/'joint',a.joint_weight,a.joint_anchors,
                        a.joint_pool,a.joint_balanced_evidence,True,a.snapshot_records,a.learning_rate)
                    runtime=batch_root/'joint/runtime'
                    verify_joint(a.root,batch_root/'joint')
                else:
                    export(a.root,checkpoint,batch_root/'runtime',learning_rate=a.learning_rate,retained=True,
                           retained_settle=True,context_length=16,optimization='O2',snapshot_records=a.snapshot_records,
                           corpus_records=batch_root/'input/records.jsonl',corpus_entry=True,graph_library=a.graph_library,graph_vocabulary=a.graph_vocabulary,optimizer=a.optimizer)
                    runtime=batch_root/'runtime'
            profile=runtime/'corpus_profile'
            command=[str(a.root/'Vox/target/release/vox'),'profile-native',str(runtime/'model'),str(profile),
                     '--registers','--duration-ms',str(a.duration_ms)]
            state.update(status='executing',command=command);save(state_path,state)
            with (batch_root/'profile_launcher.log').open('wb') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                state['process_pid']=child.pid;save(state_path,state)
                code=child.wait()
            if not profile.with_suffix('.status').exists():
                state.update(status='execution_error',exit_status=code);save(state_path,state);raise RuntimeError('profile did not return a status receipt')
            with (batch_root/'collection.log').open('w') as log,contextlib.redirect_stdout(log):
                if joint_binding:collect_joint(batch_root/'joint')
                else:collect(runtime,batch_root/'input',profile)
            receipt=json.loads((runtime/'corpus_training_receipt.json').read_text())
            if not receipt['records']:raise RuntimeError('no committed progress')
            checkpoint=runtime/'returned_checkpoint';cursor=checkpoint/'corpus_cursor.json'
            completed_batch={'batch':next_batch,'receipt':str(runtime/'corpus_training_receipt.json'),
                'records':receipt['records'],'tokens':receipt['tokens'],'status':receipt['status'],
                'checkpoint_sha256':receipt['returned_checkpoint_sha256']}
            if trajectory_bindings:
                state.update(status='training_graph_trajectories');save(state_path,state)
                trajectories=batch_root/'trajectories'
                with (batch_root/'trajectory_launcher.log').open('w') as log,contextlib.redirect_stdout(log):
                    if a.trajectory_backend=='cuda-composed':
                        if a.graph_reentry:
                            from train_banked_cuda_trajectories import train as train_cuda
                        else:
                            from train_cuda_trajectories import train as train_cuda
                        train_cuda(a.root,checkpoint,a.trajectory_dataset,a.graph_library,trajectories,
                            a.trajectory_count,a.trajectory_steps,a.trajectory_seed+next_batch,
                            a.cuda_devices,a.composed_passes,a.composed_local_weight,
                            a.composed_anchor_weight,a.composed_anchor_count,
                            batch_root/'input/records.jsonl',a.composed_source_weight,a.typed_tuples,
                            **({'graph_reentry':True,'graph_weight':a.composed_graph_weight,'distinction_weight':a.composed_distinction_weight,
                                'distinction_count':a.composed_distinction_count} if a.graph_reentry else {}))
                    else:
                        train_trajectories(a.root,checkpoint,a.trajectory_dataset,a.graph_library,trajectories,
                            a.trajectory_count,a.trajectory_steps,a.trajectory_seed+next_batch,a.duration_ms,mode=a.trajectory_mode)
                trajectory_receipt=json.loads((trajectories/'trajectory_training_receipt.json').read_text())
                checkpoint=trajectories/'returned_checkpoint';cursor=checkpoint/'corpus_cursor.json'
                completed_batch.update(trajectory_receipt=str(trajectories/'trajectory_training_receipt.json'),
                    trajectory_updates=trajectory_receipt['total_updates'],
                    checkpoint_sha256=trajectory_receipt['checkpoint_sha256'],
                    corpus_checkpoint_sha256=receipt['returned_checkpoint_sha256'])
            state['completed_batches'].append(completed_batch)
            next_batch+=1;launched+=1
            state.update(status='ready',checkpoint=str(checkpoint),cursor=str(cursor),next_batch=next_batch,inflight=None)
            state.pop('process_pid',None);save(state_path,state)
            print(json.dumps({'batch':next_batch-1,'records':receipt['records'],'tokens':receipt['tokens'],
                              'checkpoint':str(checkpoint),'status':receipt['status']}),flush=True)
        state['status']='batch_limit_reached';save(state_path,state)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','source','index','tasks','checkpoint','cursor','out'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('graph-records','graph-library','graph-vocabulary'):p.add_argument('--'+name,type=Path)
    p.add_argument('--checking-tasks',type=Path,help='training-only source-bound checking proposals')
    p.add_argument('--checking-queries',type=int,default=128)
    p.add_argument('--graph-queries',type=int,default=512)
    p.add_argument('--batches',type=int,default=0,help='zero continues until all indexed corpus records have trained')
    p.add_argument('--token-budget',type=int,default=64000);p.add_argument('--max-records',type=int,default=64)
    p.add_argument('--oracle-queries',type=int,default=128);p.add_argument('--snapshot-records',type=int,default=16)
    p.add_argument('--duration-ms',type=int,default=300000);p.add_argument('--learning-rate',type=float,default=.001)
    p.add_argument('--optimizer',choices=('sgd','adam'),default='sgd')
    p.add_argument('--trajectory-dataset',type=Path,help='train-only complete graph trajectories after each returned corpus batch')
    p.add_argument('--trajectory-count',type=int,default=32)
    p.add_argument('--trajectory-mode',type=int,choices=(2,3),default=2,help='propagation mode for the native trajectory backend')
    p.add_argument('--trajectory-steps',type=int,default=1)
    p.add_argument('--trajectory-seed',type=int,default=83)
    p.add_argument('--trajectory-backend',choices=('native','cuda-composed'),default='native',help='native local corrections or CUDA whole-graph gradients')
    p.add_argument('--cuda-devices',nargs='+',default=['cuda:0','cuda:1'])
    p.add_argument('--composed-passes',type=int,default=2)
    p.add_argument('--composed-local-weight',type=float,default=.25)
    p.add_argument('--composed-graph-weight',type=float,default=1.,help='graph objective contribution; zero retains executed graph state while language drives updates')
    p.add_argument('--composed-anchor-weight',type=float,default=0.)
    p.add_argument('--composed-anchor-count',type=int,default=8)
    p.add_argument('--composed-source-weight',type=float,default=0.,help='complete source-record rehearsal within each composed GPU gradient')
    p.add_argument('--composed-distinction-weight',type=float,default=0.,help='independent support/refutation policy loss from executing distinction oracles in each banked GPU update')
    p.add_argument('--composed-distinction-count',type=int,default=8)
    p.add_argument('--typed-tuples',action='store_true',help='axis-typed tuple objective inside shared CUDA source/graph updates')
    p.add_argument('--graph-reentry',action='store_true',help='re-enter source-bound continuous memory on later graph visits')
    p.add_argument('--joint-graph-source',type=Path,help='training-only graph records for a joint loss within each language update')
    p.add_argument('--joint-weight',type=float,default=1.)
    p.add_argument('--joint-anchors',type=int,default=8)
    p.add_argument('--joint-pool',type=int,default=128)
    p.add_argument('--joint-balanced-evidence',action='store_true')
    a=p.parse_args()
    for name in ('root','source','index','tasks','checkpoint','cursor','out'):setattr(a,name,getattr(a,name).resolve())
    graph_values=[getattr(a,name) for name in ('graph_records','graph_library','graph_vocabulary')]
    if any(graph_values) and not all(graph_values):p.error('graph records, library, and vocabulary are required together')
    for name in ('graph_records','graph_library','graph_vocabulary'):
        if getattr(a,name):setattr(a,name,getattr(a,name).resolve())
    if a.checking_tasks:a.checking_tasks=a.checking_tasks.resolve()
    if a.trajectory_dataset:
        a.trajectory_dataset=a.trajectory_dataset.resolve()
        if not a.graph_library or a.optimizer!='adam':p.error('graph trajectories require the linked graph oracle and Adam continuation')
        if a.trajectory_count<1 or not 0<=a.trajectory_steps<=100:p.error('positive trajectory count and zero to 100 correction steps required')
    if a.trajectory_backend=='cuda-composed':
        if not a.trajectory_dataset:p.error('CUDA graph training requires a trajectory dataset')
        if not a.cuda_devices or len(set(a.cuda_devices))!=len(a.cuda_devices) or any(not device.startswith('cuda:') for device in a.cuda_devices):p.error('distinct CUDA device names required')
        if not 1<=a.composed_passes<=16 or not 0<=a.composed_local_weight<=100 or not 0<=a.composed_anchor_weight<=100 or a.composed_anchor_count<1:p.error('invalid composed graph or rehearsal bounds')
        if a.composed_anchor_weight and a.composed_anchor_count%2:p.error('balanced oracle rehearsal requires an even anchor count')
        if not 0<=a.composed_source_weight<=100:p.error('source rehearsal weight must be between zero and 100')
    if a.joint_graph_source:
        a.joint_graph_source=a.joint_graph_source.resolve()
        if not a.graph_library or a.optimizer!='adam':p.error('joint graph gradients require the linked graph oracle and Adam continuation')
        if not 0<=a.joint_weight<=100 or a.joint_anchors<1 or a.joint_pool<a.joint_anchors:p.error('invalid joint-gradient weight or source bank size')
    if a.batches<0 or a.snapshot_records<1 or a.duration_ms<1:p.error('positive snapshot interval and duration, nonnegative batch limit required')
    run(a)
