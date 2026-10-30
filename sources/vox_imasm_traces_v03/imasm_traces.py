#!/usr/bin/env python3
"""Capture real IMASM CLI evidence for a validated imscription corpus.

Never infer logical verdicts from process exit codes or fabricated traces.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from bootstrap import AXES, encode_address, split_for_address


def invoke(exe, args, timeout):
    try:
        p = subprocess.run([exe, *args], capture_output=True, text=True,
                           encoding='utf-8', errors='replace', timeout=timeout,
                           stdin=subprocess.DEVNULL, check=False)
        return {'args': args, 'status': 'ok' if p.returncode == 0 else 'failed',
                'exit_code': p.returncode, 'stdout': p.stdout[:200000],
                'stderr': p.stderr[:200000]}
    except subprocess.TimeoutExpired:
        return {'args': args, 'status': 'timeout', 'timeout_seconds': timeout}
    except OSError as exc:
        return {'args': args, 'status': 'launch_error', 'error': str(exc)}


def capture(objects, output, exe='imasm', limit=10, timeout=10, per_object=False,
            prove=False, resume=False):
    if not shutil.which(exe) and not Path(exe).is_file():
        raise FileNotFoundError(f'Cannot locate IMASM: {exe}')
    output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in objects.read_text(encoding='utf8').splitlines() if line]
    if limit is not None: rows = rows[:limit]
    path = output/'traces.jsonl'
    seen = set()
    if resume and path.exists():
        for line in path.read_text(encoding='utf8').splitlines():
            try: seen.add(json.loads(line)['id'])
            except (ValueError, KeyError): pass
    type_map_file = output/'type_map.json'
    type_map = json.loads(type_map_file.read_text(encoding='utf8')) if type_map_file.exists() else {}
    written = failures = 0
    with path.open('a' if resume else 'w', encoding='utf8') as f:
        def emit(record):
            nonlocal written, failures
            if record['id'] in seen: return
            f.write(json.dumps(record, ensure_ascii=False)+'\n'); f.flush()
            written += 1
            failures += int(any(command['status'] != 'ok' for command in record.get('commands',{}).values()))
        emit({'id':'registry:types', 'kind':'type_registry',
              'commands': {'types': invoke(exe,['types'],timeout)}})
        for _, _, alphabet in AXES:
            for glyph in alphabet:
                name=type_map.get(glyph)
                emit({'id':'type:'+glyph,'kind':'type_expansion','glyph':glyph,
                      'type_name':name,'status':'captured' if name else 'needs_type_map',
                      'commands':{'expand':invoke(exe,['expand',name],timeout)} if name else {}})
        if per_object:
            for row in rows:
                name, address = row['name'], row['address']
                ident = f"object:{row['source_line']}:" + hashlib.sha256((name+address).encode()).hexdigest()[:16]
                commands={'check': invoke(exe,['check',address],timeout),
                          'eval': invoke(exe,['eval',name,'seed=B'],timeout),
                          'eval16': invoke(exe,['eval16',name,'seed=A'],timeout)}
                if prove: commands['prove']=invoke(exe,['prove',name],timeout)
                emit({'id':ident,'kind':'object_execution','name':name,
                      'address':address,'crystal_index':encode_address(address),
                      'split':split_for_address(address),'source_line':row['source_line'],
                      'grammar_judgment':None,'kernel_judgment':None,
                      'flow_judgment':None,'sixteen3_transitions':None,
                      'commands':commands})
    report={'new_records':written,'new_records_with_command_failures':failures,
            'objects_considered':len(rows),'type_mappings_supplied':len(type_map),
            'per_object':per_object,'evidence_file':str(path),
            'notice':'Raw CLI output only; no inferred proof verdicts or internal flow steps.'}
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('objects',type=Path)
    p.add_argument('--out',type=Path,default=Path('traces'))
    p.add_argument('--imasm',default='imasm')
    p.add_argument('--limit',type=int,default=10)
    p.add_argument('--timeout',type=float,default=10)
    p.add_argument('--per-object',action='store_true')
    p.add_argument('--prove',action='store_true')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.limit < 0 or a.timeout <= 0: p.error('limit must be >= 0 and timeout > 0')
    if a.prove and not a.per_object: p.error('--prove requires --per-object')
    print(json.dumps(capture(a.objects,a.out,a.imasm,a.limit,a.timeout,a.per_object,a.prove,a.resume),indent=2))

if __name__=='__main__': main()
