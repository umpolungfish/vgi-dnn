#!/usr/bin/env python3
"""Expose existing complete corpus instructions as source-bound dialogue turns."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def convert(row):
    if row['kind'] != 0 or row.get('training_split') != 'train':
        raise ValueError('a complete training corpus record is required')
    if row['prefix'][0] != 256 or row['prefix'][-1] != 257 or row['target'][-1] != 258:
        raise ValueError('source framing changed')
    prompt = bytes(row['prefix'][1:-1])
    target = bytes(row['target'][:-1])
    if digest(prompt) != row['prompt_sha256'] or digest(target) != row['target_sha256']:
        raise ValueError('source prompt or continuation binding changed')
    dialogue = 'User: ' + prompt.decode('utf8') + '\nAssistant:'
    prefix = [256] + list(dialogue.encode()) + [257]
    source = dict(row)
    source.update(example_id=row['example_id'] + ':dialogue', prefix=prefix,
                  prompt_sha256=digest(dialogue.encode()),
                  dialogue_provenance={'original_record': row,
                                       'original_record_sha256': digest(json.dumps(row, sort_keys=True).encode())})
    if row['task'] in ('catalog_to_tuple', 'ob3ect_to_tuple'):
        # These continue to execute the complete-glyph axis objective.
        source['dialogue_objective'] = 'typed tuple; preserve axis-bound operation'
        return source
    source.update(kind=6, task='paired_language_evidence', corpus_task=row['task'],
                  prompt=dialogue, target=[], support_target=target.decode('utf8'),
                  support_sha256=digest(target), refutation_target=None, refutation_sha256=None,
                  evidence_bits=1, evidence_presence=True, presence_mask=[1,0],
                  evidence_bits_role='supervised continuation channels; refutation remains unobserved',
                  dialogue_objective='attested source continuation through affirmative language readout')
    return source


def build(records, out):
    source = [json.loads(line) for line in records.open()]
    rows = [convert(row) for row in source if row['kind'] == 0]
    if not rows:
        raise ValueError('no complete source language records')
    out.mkdir(parents=True, exist_ok=False)
    (out/'records.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows))
    report = {'source':str(records),'source_sha256':digest(records.read_bytes()),
              'records':len(rows),'typed_records':sum(row['kind']==0 for row in rows),
              'tasks':sorted({row.get('corpus_task',row['task']) for row in rows}),
              'complete_targets_preserved':True,'refutation':'unobserved and unlabelled',
              'scope':'dialogue framing of existing complete corpus records; no new answers or source assertions'}
    (out/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    build(args.records.resolve(),args.out.resolve())
