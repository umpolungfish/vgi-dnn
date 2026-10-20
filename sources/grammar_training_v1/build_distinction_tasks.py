#!/usr/bin/env python3
"""Reference choices among IMASM constructions with co-present evidence.

Targets describe every candidate's observed return under a declared criterion.
Several candidates may receive support; no single distinction is canonicalized.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

from bootstrap import split_for_address


def build(root,output):
    output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(root/'VGI/sources/GRAMMAR_UNIFIED_RUN'))
    from carrier_transitions import gate,split,fuse
    operations=list('≻≺⋈⊤⊙⊥⊞⊡')
    templates=[]
    for action,arity in [('direct',0),('split_two',2),('split_three',3),('release',0)]:
        nodes=[{'id':'source','opcode':'⊢'},{'id':'return','opcode':'⊣'}]
        edges=[]
        if arity:
            nodes.extend([{'id':'fork','opcode':'∈','arity':arity},{'id':'fuse','opcode':'∋','arity':arity}])
            edges.extend([['source',0,'fork',0],['fuse',0,'return',0]])
            for arm in range(arity):
                node=f'arm_{arm}';nodes.append({'id':node,'opcode':'$work' if arm==0 else '≻'})
                edges.extend([['fork',arm,node,0],[node,0,'fuse',arm]])
        else:
            nodes.append({'id':'work','opcode':'⊙' if action=='release' else '$work'})
            edges.extend([['source',0,'work',0],['work',0,'return',0]])
        templates.append({'action':action,'arity':arity,'nodes':nodes,'edges':edges,
                          'returned_view':arity})
    (output/'graph_templates.json').write_text(json.dumps(templates,ensure_ascii=False,indent=2)+'\n')
    counts=Counter();shapes=Counter();multi=0
    handles={s:(output/f'{s}.jsonl').open('w',encoding='utf8') for s in ('train','validation','test')}
    try:
        for seed in range(16):
            for op_index,work in enumerate(operations):
                candidates=[]
                for template in templates:
                    arity=template['arity']
                    if arity:
                        before=list(split(seed,arity));after=before.copy();after[0]=gate(work,after[0]);value=fuse(after)
                    else:
                        before=after=[];value=seed if template['action']=='release' else gate(work,seed)
                    candidates.append({'action':template['action'],'return_bits':value,
                                       'view':arity,'arms_before':before,'arms_after':after})
                family=f'seed={seed};work={work}';partition=split_for_address(family)
                for positive in sorted({c['return_bits'] for c in candidates}):
                    for negative in (None,positive,positive^15):
                        for previous in range(4):
                            for view in (None,0,2,3):
                                support=[int(c['return_bits']==positive and (view is None or c['view']==view)) for c in candidates]
                                refute=[int((negative is not None and c['return_bits']==negative) or
                                            (view is not None and c['view']!=view)) for c in candidates]
                                row={'id':hashlib.sha256(json.dumps([family,positive,negative,previous,view],ensure_ascii=False).encode()).hexdigest(),
                                     'family':family,'split':partition,'seed_bits':seed,'work':work,'work_index':op_index,
                                     'positive_return':positive,'negative_return':negative,'previous_action':previous,
                                     'requested_view':view,'candidates':candidates,
                                     'support':support,'refute':refute,'evidence':'reference_counterfactual_graph_returns'}
                                handles[partition].write(json.dumps(row,ensure_ascii=False)+'\n');counts[partition]+=1
                                multi+=sum(support)>1
                                shapes.update(('N','T','F','B')[t+2*f] for t,f in zip(support,refute))
    finally:
        for handle in handles.values():handle.close()
    source=root/'VGI/sources/GRAMMAR_UNIFIED_RUN/carrier_transitions.py'
    report={'counts':dict(counts),'candidate_evidence_shapes':dict(shapes),
            'multiple_supported_constructions':multi,'operations':operations,
            'actions':[t['action'] for t in templates],
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'selection_rule':'learned policy chooses; exact candidate outcomes are evaluation/training evidence',
            'view_rule':'an optional observation criterion for this query; no persistent mandatory distinction',
            'split_grouping':'seed/work family, including all goals, previous actions, and views',
            'status':'reference task data; policy training and executing choice verification pending'}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();build(a.root.resolve(),a.out.resolve())
