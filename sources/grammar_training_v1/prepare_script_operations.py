#!/usr/bin/env python3
"""Teach source-bound writing operations returned by the existing script compilers."""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def package(root, folder, name):
    path = root / folder / 'programs'
    spec = importlib.util.spec_from_file_location(name, path / '__init__.py', submodule_search_locations=[str(path)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return __import__(name + '.compiler', fromlist=['compiler'])


def build(root, out):
    if out.exists():
        raise FileExistsError(out)
    rows = []
    for folder, corpus, name in [('Linear_Analytica','linear_a_latff_sample.txt','linear_operations'),
                                  ('Rongorongo_Tau','barthel_corpus.txt','tau_operations')]:
        compiler = package(root, folder, name)
        source = root / folder / 'data' / corpus
        text = source.read_text()
        compiled = (compiler.compile_corpus(source, workers=1) if folder == 'Linear_Analytica'
                    else compiler.compile_transcription(source))
        sections, current = {}, None
        for line in text.splitlines():
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            if folder == 'Linear_Analytica' and '<t' in line:
                current = 't' + line.split('<t')[1].split('>')[0].split('.')[0]
                sections.setdefault(current, [])
            elif folder == 'Rongorongo_Tau' and line.startswith('>'):
                current = line[1:].strip()
                sections.setdefault(current, [])
            elif current is not None:
                sections[current].append(line)
        outputs = compiled.get('pages', compiled.get('tablets'))
        bindings = {str(p.relative_to(root)):sha(p.read_bytes()) for p in
                    [source, root/folder/'programs/compiler.py', root/folder/'programs/primitives.py']}
        for section, lines in sections.items():
            result = outputs[section]
            prompt = ('User: Return the Grammar operation stream produced by the existing '
                      + folder + ' compiler for this transcription. Preserve raw data and line direction.\n'
                      + '\n'.join(lines) + '\nAssistant:')
            target = '\n'.join(result['instructions'])
            identity = folder + ':' + section
            rows.append(dict(kind=6,task='paired_language_evidence',id=identity,example_id=identity,
                source_id=identity,family='script_operations',training_split='train',prompt=prompt,
                prefix=[256]+list(prompt.encode())+[257],target=[],prompt_sha256=sha(prompt.encode()),
                support_target=target,support_sha256=sha(target.encode()),refutation_target=None,
                refutation_sha256=None,evidence_bits=1,evidence_presence=True,presence_mask=[1,0],
                source={'path':str(source),'sha256':sha(source.read_bytes())},
                compiler_binding=bindings,compiler_return=result,
                evidence_scope='Existing compiler return; refutation unobserved.'))
    out.mkdir(parents=True)
    (out/'records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    report={'records':len(rows),'source_folders':['Linear_Analytica','Rongorongo_Tau'],
            'records_sha256':sha((out/'records.jsonl').read_bytes()),
            'scope':'Complete source sections paired with actual compiler returns; no natural-language decipherment labels.'}
    (out/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('.'))
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    build(args.root.resolve(),args.out.resolve())
