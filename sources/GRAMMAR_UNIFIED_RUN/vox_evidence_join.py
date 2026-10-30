#!/usr/bin/env python3
"""Join observed Vox verdicts with immutable named IMASM probes and export training evidence."""
import argparse
import hashlib
import json
from pathlib import Path

VALID = {'T', 'F', 'B', 'N', '-'}

def join(report_path, destination):
    report = json.loads(Path(report_path).read_text(encoding='utf-8'))
    if report.get('status') != 'completed' or report.get('returncode') != 0:
        raise ValueError('Native verdict collection must complete successfully')
    probes = report['probes']
    names = [row['name'] for row in probes]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate probe names')
    by_name = {row['name']: row for row in probes}
    parsed = {}
    for lineno, raw in enumerate(report.get('stdout', '').splitlines(), 1):
        if not raw.strip():
            continue
        parts = raw.split('\t')
        if len(parts) != 3:
            raise ValueError(f'Invalid Vox output at line {lineno}: expected 3 TSV fields')
        name, verdict, length = parts
        if name not in by_name or name in parsed:
            raise ValueError(f'Unexpected or repeated Vox label: {name}')
        if verdict not in VALID or not length.isdecimal():
            raise ValueError(f'Invalid Vox verdict or length for {name}: {verdict}, {length}')
        if int(length) != len(by_name[name]['word']):
            raise ValueError(f'Verdict length mismatch for {name}')
        parsed[name] = {'verdict': verdict, 'glyph_count': int(length)}
    if set(parsed) != set(names):
        raise ValueError(f'Incomplete native evidence: missing {sorted(set(names)-set(parsed))}')
    rows=[]
    for p in probes:
        name, word=p['name'],p['word']
        rows.append({'kind':'native_vox_verdict','name':name,'word':word,
                     'word_sha256':hashlib.sha256(word.encode()).hexdigest(),
                     'verdict':parsed[name]['verdict'],'glyph_count':parsed[name]['glyph_count'],
                     'observation':'vox verdict --tsv',
                     'source':'observed_native_execution'})
    dest=Path(destination)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(''.join(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n' for row in rows),encoding='utf-8')
    summary={'native_samples':len(rows), 'verdict_counts':{v:sum(r['verdict']==v for r in rows) for v in ('T','F','B','N','-')},
             'training_records':str(dest),'transition_pairs':0}
    (dest.parent/'native_join_report.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print('NATIVE_JOIN',json.dumps(summary))
    return rows

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--evidence',default='native_vox_evidence.json')
    parser.add_argument('--output',default='native_verdict_training.jsonl')
    args=parser.parse_args()
    join(args.evidence,args.output)
