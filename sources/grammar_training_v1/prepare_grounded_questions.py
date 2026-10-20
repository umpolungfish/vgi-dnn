#!/usr/bin/env python3
"""Frame exact canonical source paragraphs as direct language questions."""
import hashlib
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def build():
    root=Path(__file__).resolve().parents[3]
    source=root/'ig-docs/The_Dialetheic_Neural_Network.md'
    text=source.read_text()
    requests=[('What is a dialetheic neural network?', '## Abstract'),
              ('What is the Imscribing Grammar?', '## 2. Ontological premise: process first, distinction available'),
              ('What does Both mean?', '### 3.2 Both is information, not explosion')]
    rows=[]
    for i,(question,heading) in enumerate(requests):
        section=text.split(heading+'\n',1)[1].lstrip('\n')
        paragraph=section.split('\n\n',1)[0]
        assert paragraph and paragraph in text and not paragraph.startswith('#')
        prompt='User: '+question+'\nAssistant:'
        rows.append(dict(kind=6,task='paired_language_evidence',id=f'grounded-question:{i}',
            example_id=f'grounded-question:{i}',source_id=str(source.relative_to(root))+':'+heading,
            family='canonical_source_question',training_split='train',prompt=prompt,
            prefix=[256]+list(prompt.encode())+[257],target=[],prompt_sha256=digest(prompt.encode()),
            support_target=paragraph,support_sha256=digest(paragraph.encode()),refutation_target=None,
            refutation_sha256=None,evidence_bits=1,evidence_presence=True,presence_mask=[1,0],
            source={'path':str(source),'sha256':digest(source.read_bytes())},
            passage={'heading':heading,'byte_start':text.encode().index(paragraph.encode()),
                     'byte_length':len(paragraph.encode())},
            evidence_scope='Verbatim canonical source paragraph; refutation unobserved.'))
    out=root/'VGI/grounded_questions_20261017'
    out.mkdir(exist_ok=False)
    p=out/'records.jsonl';p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    (out/'preparation.json').write_text(json.dumps({'records':len(rows),'records_sha256':digest(p.read_bytes()),
       'scope':'Direct questions with complete verbatim source paragraphs. Complementary dialogue exposure; no generated factual answers.'},indent=2)+'\n')
    print('Prepared direct questions with exact canonical source passages.')


if __name__=='__main__':
    build()
