#!/usr/bin/env python3
"""SIXTEEN_3 split/AREV/fuse data, with optional independent native CLI evidence.

Carrier is the power set of {T,F,t,f}; masks bit 0..3. No invented VM verdicts.
For two-arm splitting the documented truth/falsity partition is {T,t}/{F,f}.
For three-arm splitting it is {T}/{F}/{t,f}.
"""
import argparse, itertools, json, random, shutil, subprocess
from pathlib import Path

BITS = ('T','F','t','f')
TRUTH = 0b0101
FALSITY = 0b1010
AREV = (1,0,3,2)

def display(mask):
    if mask == 0: return 'N'
    if mask == 1: return 'T'
    if mask == 2: return 'F'
    if mask == 3: return 'B'
    return ''.join(v for i,v in enumerate(BITS) if mask & (1<<i))

def reverse(mask):
    return sum((1<<AREV[i]) for i in range(4) if mask & (1<<i))

def split(mask, arms=2):
    if arms == 2: return (mask & TRUTH, mask & FALSITY)
    if arms == 3: return (mask & 1, mask & 2, mask & 12)
    raise ValueError('arity must be 2 or 3')

def evaluate(mask, arms, transforms):
    before=split(mask,arms)
    after=tuple(reverse(b) if flag else b for b,flag in zip(before,transforms))
    fused=0
    for b in after: fused |= b
    return dict(seed=display(mask),seed_mask=mask,arity=arms,
        transforms=['AREV' if t else 'HOLD' for t in transforms],
        split_masks=list(before),split_values=[display(m) for m in before],
        transformed_masks=list(after),fused_mask=fused,fused=display(fused),
        recovered=(fused == mask), flow_identity=(fused == mask))

def native_capture(exe,word,seed,timeout):
    # The exact native protocol may differ across installed versions.
    # Capture raw evidence without interpreting verdicts or asserting parity.
    results={}
    for verb in ('check','eval16'):
        args=[exe,verb,word] + ([f'seed={seed}'] if verb=='eval16' else [])
        try:
            p=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
            results[verb]={'command':args,'returncode':p.returncode,
                            'stdout':p.stdout,'stderr':p.stderr}
        except (OSError,subprocess.TimeoutExpired) as exc:
            results[verb]={'command':args,'error':str(exc)}
    return results

def write_dataset(out,seed=42,limit_native=0,exe=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    records=[]
    for mask in range(16):
        for arity in (2,3):
            for flags in itertools.product((False,True),repeat=arity):
                e=evaluate(mask,arity,flags)
                e['evidence']='specified_carrier_algebra'
                e['grammar_judgment']=None
                e['kernel_judgment']=None
                e['native_flow_judgment']=None
                records.append(e)
    # Keep all 192 truth-table rows as the definitive spec test corpus.
    # Assign split by *seed* to prevent same seed leaking into different partitions.
    rng=random.Random(seed); masks=list(range(16));rng.shuffle(masks)
    train=set(masks[:12]);val=set(masks[12:14]);test=set(masks[14:])
    for name,selection in [('train',train),('validation',val),('test',test)]:
        with (out/f'{name}.jsonl').open('w',encoding='utf8') as f:
            for item in records:
                if item['seed_mask'] in selection: f.write(json.dumps(item,ensure_ascii=False)+'\n')
    native=[]
    if limit_native:
        if not exe or not shutil.which(exe):
            raise FileNotFoundError('Native IMASM unavailable. Pass --imasm /path/to/imasm, or omit --native-probes. Generated spec examples are not VM execution evidence.')
        # Real native check/eval16 queries, kept distinct from the algebraic oracle.
        word='⊢∈≻∋⊤'
        for row in records[:limit_native]:
            native.append({'spec_case':row,'native':native_capture(exe,word,row['seed'],5),
                           'word':word,'note':'Native protocol probe; equivalence to masked arm transform not assumed'})
        with (out/'native_probes.jsonl').open('w',encoding='utf8') as f:
            for row in native:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    report={'cases':len(records),'split_by_seed':True,
            'splits':{n:sum(r['seed_mask'] in group for r in records)
                    for n,group in [('train',train),('validation',val),('test',test)]},
            'identity_cases':sum(r['recovered'] for r in records),
            'non_identity_cases':sum(not r['recovered'] for r in records),
            'native_probes':len(native),'provenance':'IMSCRIBERS_GUIDE_TO_IMASM.md, Part IV',
            'asserted_grammar_kernel_verdicts':False}
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='dynamic_training')
    p.add_argument('--native-probes',type=int,default=0)
    p.add_argument('--imasm',default='imasm')
    a=p.parse_args();print(json.dumps(write_dataset(a.out,limit_native=a.native_probes,exe=a.imasm),indent=2))
if __name__=='__main__': main()
