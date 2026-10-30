#!/usr/bin/env python3
"""Capture native Vox verdict evidence using the CLI's required TSV file argument."""
import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_WORDS = [
    ('sourceWord', '⊢≻⊤⋈∈⊤⊥⊞∋≺⊙⊡⊣'),
    ('strandedWord', '⊢∋⊤⋈∈⊤⊥⊞≻≺⊙⊡⊣'),
]

def probes(path=None):
    if path is None:
        return DEFAULT_WORDS
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip() or line.startswith('#'):
            continue
        cols = line.split('\t')
        if len(cols) != 2 or not cols[0] or not cols[1]:
            raise ValueError(f'Invalid name<TAB>word on line {line_number}')
        rows.append(tuple(cols))
    if not rows:
        raise ValueError('No IMASM words supplied')
    return rows

def run(vox='vox', words=None, output='native_vox_evidence.json', timeout=30):
    records = probes(words)
    data = ''.join(f'{n}\t{w}\n' for n, w in records)
    executable = shutil.which(vox) if '/' not in vox else (vox if Path(vox).is_file() else None)
    report = {
        'captured_utc': datetime.now(timezone.utc).isoformat(),
        'command_pattern': 'vox verdict --tsv <file>',
        'probe_count': len(records),
        'input_sha256': hashlib.sha256(data.encode('utf-8')).hexdigest(),
        'probes': [{'name': n, 'word': w} for n, w in records],
        'interface': 'vox verdict --tsv <file>',
        'source': 'observed CLI execution',
    }
    if not executable:
        report.update(status='unavailable', reason='vox executable not found', returncode=None, stdout='', stderr='')
    else:
        with tempfile.TemporaryDirectory(prefix='grammar_vox_') as directory:
            path = Path(directory) / 'probes.tsv'
            path.write_text(data, encoding='utf-8')
            try:
                proc = subprocess.run([str(executable), 'verdict', '--tsv', str(path)],
                                      capture_output=True, text=True, encoding='utf-8',
                                      errors='replace', timeout=timeout, check=False)
                report.update(status='completed' if proc.returncode == 0 else 'command_failed',
                              returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
            except subprocess.TimeoutExpired as exc:
                report.update(status='timeout', returncode=None,
                              stdout=(exc.stdout or b'').decode('utf-8', 'replace') if isinstance(exc.stdout, bytes) else str(exc.stdout or ''),
                              stderr=(exc.stderr or b'').decode('utf-8', 'replace') if isinstance(exc.stderr, bytes) else str(exc.stderr or ''))
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('VOX_NATIVE', json.dumps({
        'status': report['status'], 'probes': len(records),
        'returncode': report.get('returncode'), 'output': str(output),
        'stderr': report.get('stderr', '')[:500] if report['status'] != 'completed' else '',
    }, ensure_ascii=False))
    return report

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--vox', default='vox')
    p.add_argument('--words', help='Optional TSV file name<TAB>IMASM word')
    p.add_argument('--output', default='native_vox_evidence.json')
    p.add_argument('--timeout', type=int, default=30)
    a = p.parse_args()
    result = run(a.vox, a.words, a.output, a.timeout)
    raise SystemExit(0 if result['status'] in ('completed', 'unavailable') else 1)
