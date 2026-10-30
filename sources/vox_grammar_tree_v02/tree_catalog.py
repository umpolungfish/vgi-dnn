#!/usr/bin/env python3
"""Compile a full imscription tree+table export without truncated console names."""
import argparse, collections, json, re
from pathlib import Path
from bootstrap import compile_catalog, parse_catalog

TREE_LINE = re.compile(r'^(?P<prefix>[│ ]*)(?:├── |└── )(?P<entry>.+)$')
META = re.compile(r'^(?P<name>.*?) \(F:(?P<fidelity>[^,]+), P:(?P<parity>[^)]+)\)$')

def tree_metadata(text):
    metadata=collections.defaultdict(list)
    ancestry=[]
    for lineno, line in enumerate(text.splitlines(),1):
        if line.startswith('9432 imscriptions displayed.'):
            break
        m=TREE_LINE.match(line)
        if not m: continue
        depth=len(m.group('prefix'))//4
        value=m.group('entry').strip()
        ancestry=ancestry[:depth]
        if depth==0:
            ancestry=[value]
            continue
        item=META.match(value)
        if item:
            metadata[item['name']].append({'tree_path':list(ancestry),'fidelity_name':item['fidelity'],
                                           'parity_name':item['parity'],'tree_line':lineno})
        else:
            ancestry.append(value)
    return metadata

def compile_tree(source, output):
    text=source.read_text(encoding='utf-8')
    marker='Imscriptions — 1–8777 of 8777'
    if marker not in text:
        raise ValueError('full tabular catalog section not found')
    table=text.split(marker,1)[1]
    # Compile the canonical full names/addresses from the table alone.
    output.mkdir(parents=True,exist_ok=True)
    tmp=output/'_extracted_table.txt'
    tmp.write_text(table,encoding='utf-8')
    report=compile_catalog(tmp,output)
    tmp.unlink()
    metadata=tree_metadata(text)
    matches=0
    path_missing=0
    augmented=[]
    with (output/'objects.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            m=metadata.get(r['name'],[])
            if m: matches+=1
            else: path_missing+=1
            r['tree_occurrences']=m
            r['source']='grammartree_full_table'
            augmented.append(r)
    with (output/'objects.jsonl').open('w',encoding='utf-8') as f:
        for r in augmented: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    report.update({'tree_names':len(metadata),'objects_with_tree_metadata':matches,
                   'objects_without_tree_metadata':path_missing,
                   'tree_occurrences':sum(len(x) for x in metadata.values())})
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('catalog',type=Path)
    p.add_argument('--out',type=Path,default=Path('compiled'))
    a=p.parse_args()
    print(json.dumps(compile_tree(a.catalog,a.out),ensure_ascii=False,indent=2))
