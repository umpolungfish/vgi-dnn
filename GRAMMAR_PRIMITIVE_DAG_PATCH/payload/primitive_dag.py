#!/usr/bin/env python3
"""Source-grounded 49-value IMASM expansion registry with shared word DAG.

Type-to-word expansions originate in SNS_PRIME.md; source opcode semantics
remain in IMSCRIBERS_GUIDE_TO_IMASM.md. A word alone does not fix graph wiring.
"""
import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

ALPHABET = '⊢⊣≻≺⋈⊤∈∋⊙⊥⊞⊡'
AXES = ('⊢', '⊣', '≻', '≺', '⋈', '⊤', '∈', '∋', '⊙', '⊥', '⊞', '⊡')
RADICES = (4, 5, 4, 5, 3, 5, 3, 4, 5, 4, 3, 4)
OPERATIONS = ('VINIT', 'TANCH', 'AFWD', 'AREV', 'CLINK', 'EVALT', 'FSPLIT', 'FFUSE', 'IMSCRIB', 'EVALF', 'ENGAGR', 'IFIX')
ENTRY = re.compile(r'^\|\s*`([^`]+)`\s*\|\s*(U\+[0-9A-Fa-f]+)\s*\|\s*([^|]+?)\s*\|\s*([⊢⊣≻≺⋈⊤∈∋⊙⊥⊞⊡]+)\s*\|?\s*$')
HEAD = re.compile(r'^####\s+([⊢⊣≻≺⋈⊤∈∋⊙⊥⊞⊡])\s+—')

def load_registry(source):
    axis = None
    entries = []
    for lineno, line in enumerate(Path(source).read_text(encoding='utf-8').splitlines(), 1):
        m = HEAD.match(line)
        if m:
            axis = m.group(1)
            continue
        m = ENTRY.match(line)
        if not m or axis is None:
            continue
        glyph, cp, name, word = m.groups()
        if ord(glyph) != int(cp[2:], 16):
            raise ValueError(f'Codepoint mismatch at line {lineno}: {glyph}')
        if any(ch not in ALPHABET for ch in word):
            raise ValueError(f'Non-IMASM glyph at line {lineno}')
        entries.append({'axis': axis, 'glyph': glyph, 'name': name.strip(), 'word': word, 'line': lineno})
    by_axis = {a: [] for a in AXES}
    for e in entries:
        by_axis[e['axis']].append(e)
    if tuple(map(lambda a: len(by_axis[a]), AXES)) != RADICES:
        raise ValueError('Incorrect radix distribution: '+str({a:len(v) for a,v in by_axis.items()}))
    if len({e['glyph'] for e in entries}) != 49:
        raise ValueError('Expected 49 distinct type glyphs')
    return entries

class WordDAG:
    def __init__(self, entries):
        self.entries = {e['glyph']: e for e in entries}
        self.nodes = {}
        self.word_roots = {}
        for e in entries:
            self.word_roots[e['glyph']] = self._intern(e['word'])

    def _intern(self, word):
        """Intern ordered strings as a balanced binary rope DAG."""
        if len(word) == 1:
            key = 'op:'+word
            self.nodes.setdefault(key, {'kind':'opcode','glyph':word})
            return key
        half = len(word)//2
        children = [self._intern(word[:half]), self._intern(word[half:])]
        key = 'seq:'+hashlib.sha256(json.dumps(children,ensure_ascii=False).encode()).hexdigest()[:24]
        self.nodes.setdefault(key, {'kind':'sequence','children':children})
        return key

    def unfold(self, root):
        node = self.nodes[root]
        if node['kind'] == 'opcode':
            return node['glyph']
        return ''.join(self.unfold(c) for c in node['children'])

    def tuple_word(self, glyphs):
        if len(glyphs) != 12:
            raise ValueError('A crystal tuple must have 12 glyphs')
        for i, g in enumerate(glyphs):
            if g not in self.entries or self.entries[g]['axis'] != AXES[i]:
                raise ValueError(f'Wrong glyph for axis {i}: {g}')
        return ''.join(self.entries[g]['word'] for g in glyphs)

    def report(self):
        unique = len({e['word'] for e in self.entries.values()})
        counts = collections.Counter(e['axis'] for e in self.entries.values())
        return {'type_values':len(self.entries), 'axes':len(AXES), 'opcode_alphabet':ALPHABET,
                'radices':[counts[a] for a in AXES], 'distinct_type_programs':unique,
                'shared_dag_nodes':len(self.nodes), 'root_nodes':len(set(self.word_roots.values())),
                'roundtrips_exact':all(self.unfold(self.word_roots[g]) == e['word'] for g,e in self.entries.items()),
                'program_lengths':{'min':min(map(lambda e:len(e['word']),self.entries.values())), 'max':max(map(lambda e:len(e['word']),self.entries.values()))},
                'source':'SNS_PRIME.md', 'graph_wiring':'specified separately from opcode word'}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default=str(Path(__file__).with_name('SNS_PRIME.md')))
    p.add_argument('--report',default='primitive_dag_report.json')
    p.add_argument('--tuple',help='12 glyphs in canonical order, optionally wrapped in ⟨⟩')
    p.add_argument('--value',help='Expand one Shavian value into its IMASM program')
    a=p.parse_args()
    dag=WordDAG(load_registry(a.source))
    report=dag.report()
    if a.value:
        report['value_expansion']={'glyph':a.value,'word':dag.entries[a.value]['word']}
    if a.tuple:
        report['tuple_expansion'] = dag.tuple_word(a.tuple.strip('⟨⟩'))
    Path(a.report).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('PRIMITIVE_DAG '+json.dumps(report,ensure_ascii=False))

if __name__ == '__main__':main()
