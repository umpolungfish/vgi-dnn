#!/usr/bin/env python3
"""Exact finite SIXTEEN_3 reference gate relations from IMSCRIBERS_GUIDE_TO_IMASM.

These are executable gate input/output pairs, NOT observed native Vox VM traces.
Vox's 'verdict' judges closure and is not a carrier evaluator.
"""
import argparse
import json
from pathlib import Path

ORDER=('T','F','t','f')
GLYPHS='⊢⊣≻≺⋈⊤∈∋⊙⊥⊞⊡'
OPERATIONS=dict(zip(GLYPHS,('VINIT','TANCH','AFWD','AREV','CLINK','EVALT','FSPLIT','FFUSE','IMSCRIB','EVALF','ENGAGR','IFIX')))
TRUTH=0b0101  # T,t: bits 0 and 2
FALSE=0b1010  # F,f: bits 1 and 3
INFO=0b1100   # t,f: bits 2 and 3

def label(bits):
    return ''.join(ch for i,ch in enumerate(ORDER) if bits&(1<<i)) or 'N'

def gate(glyph, x):
    """Non-branching gate on a four-bit carrier, x in [0,15]."""
    if glyph not in OPERATIONS or glyph in '∈∋⊢':
        raise ValueError('Requires a specified graph port/arity or seed')
    if glyph=='≺':
        return ((x&1)<<1)|((x&2)>>1)|((x&4)<<1)|((x&8)>>1)
    if glyph=='⊤':return x&TRUTH
    if glyph=='⊥':return x&FALSE
    if glyph=='⊞':return x|INFO
    return x

def split(x, arity):
    if arity==2:return (x&TRUTH,x&FALSE)
    if arity==3:return (x&1,x&2,x&INFO)
    raise ValueError('FSPLIT accepts only 2 or 3 arms')

def fuse(arms):
    if len(arms) not in (2,3): raise ValueError('FFUSE accepts 2 or 3 arms')
    out=0
    for arm in arms:
        if arm not in range(16):raise ValueError('Invalid carrier value')
        out |= arm
    return out

def generate():
    for glyph in GLYPHS:
        if glyph in '⊢∈∋': continue
        for before in range(16):
            after=gate(glyph,before)
            yield dict(kind='carrier_transition',evidence='reference_gate_semantics',
                opcode=glyph,operation=OPERATIONS[glyph],before_bits=before,after_bits=after,
                before=label(before),after=label(after),graph='one_input_one_output',
                source='IMSCRIBE_IMASM.md §5')
    for arity in (2,3):
        for before in range(16):
            arms=split(before,arity)
            after=fuse(arms)
            yield dict(kind='carrier_transition',evidence='reference_gate_semantics',
                opcode='∈…∋',operation='FFUSE∘FSPLIT',arity=arity,
                before_bits=before,after_bits=after,before=label(before),after=label(after),
                arm_bits=list(arms),arms=[label(a) for a in arms],
                graph='explicit_split_fuse_identity',source='IMSCRIBE_IMASM.md §2/§5')

def write(native_path, output, summary_path):
    native=Path(native_path)
    native_rows=[]
    if native.exists():
        for raw in native.read_text(encoding='utf8').splitlines():
            if raw.strip():
                row=json.loads(raw)
                if row.get('kind')=='native_vox_verdict':native_rows.append(row)
    if not native_rows:raise ValueError('No native Vox verdict evidence found; run native_verdict_join first')
    transitions=list(generate())
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in native_rows+transitions),encoding='utf8')
    summary=dict(native_verdict_observations=len(native_rows),transition_pairs=len(transitions),
                 reference_transition_pairs=len(transitions),native_vm_transition_pairs=0,
                 all_training_rows=len(native_rows)+len(transitions),
                 output=str(out),source='SIXTEEN_3 reference carrier gate semantics',
                 native_trace_status='not provided by vox verdict --tsv')
    Path(summary_path).write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
    print('CARRIER_TRANSITIONS',json.dumps(summary))
    return summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--native',default='native_verdict_training.jsonl')
    p.add_argument('--output',default='grammar_transition_training.jsonl')
    p.add_argument('--report',default='grammar_transition_report.json')
    a=p.parse_args();write(a.native,a.output,a.report)
if __name__=='__main__':main()
