"""Export and independently replay oracle-corrected transition examples.

Each example is a full pre-state and exact post-state, not just a guessed label.
"""
import json
from pathlib import Path
from grammar_ops import apply_operation, RADICES
from dynamic_bridge import evaluate
from certified_execution import frame_check


def replay(record):
    before=record['before']; target=record['oracle_target']; op=record['operation']
    assert frame_check(before['address'],before['carrier']), 'invalid pre-state'
    assert frame_check(target['address'],target['carrier']), 'invalid target'
    if record['kind']=='coordinate':
        addr=apply_operation(before['address'],op['mode'],op['axes'],op['values'])
        actual={'address':addr,'carrier':before['carrier']}
    elif record['kind']=='carrier':
        val=evaluate(before['carrier'],op['arity'],tuple(op['flags']))['fused_mask']
        actual={'address':before['address'],'carrier':val}
    else:
        raise ValueError('unknown operation')
    assert actual==target, f'oracle mismatch at case={record["case"]} step={record["step"]}'
    assert record['proposed']!=target, 'non-error in correction log'
    return True


def export_corrections(attribution, target):
    corrections=attribution['corrections']
    path=Path(target)
    # Validate every example before writing any output.
    for entry in corrections: replay(entry)
    with path.open('w',encoding='utf-8') as f:
        for entry in corrections:
            f.write(json.dumps(entry,sort_keys=True,separators=(',',':'))+'\n')
    reread=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
    assert len(reread)==len(corrections)
    for entry in reread: replay(entry)
    return {'count':len(corrections),'validated':len(reread),'path':str(path)}
