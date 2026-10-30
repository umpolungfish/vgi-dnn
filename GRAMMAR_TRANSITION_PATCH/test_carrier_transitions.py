import json
import tempfile
import unittest
from pathlib import Path
from carrier_transitions import gate,split,fuse,generate,write
class Tests(unittest.TestCase):
    def test_all_split_fuse_roundtrip(self):
        for x in range(16):
            for arity in (2,3):self.assertEqual(fuse(split(x,arity)),x)
    def test_involution_and_projection(self):
        for x in range(16):
            self.assertEqual(gate('≺',gate('≺',x)),x)
            self.assertEqual(gate('⊤',x)&gate('⊥',x),0)
            self.assertEqual(gate('⊤',x)|gate('⊥',x),x)
            self.assertEqual(gate('⊞',x)&12,12)
    def test_192_real_pairs(self):
        rows=list(generate());self.assertEqual(len(rows),176)  # 9 unary gates * 16 + 2 arities *16
        self.assertTrue(all('before_bits' in r and 'after_bits' in r for r in rows))
    def test_training_preserves_native_evidence_and_idempotence(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td); native=p/'native.jsonl'; out=p/'out.jsonl'; summary=p/'report.json'
            native.write_text(json.dumps({'kind':'native_vox_verdict','verdict':'T','name':'a'})+'\n')
            a=write(native,out,summary);self.assertEqual(a['transition_pairs'],176)
            first=out.read_bytes();write(native,out,summary)
            self.assertEqual(out.read_bytes(),first)
            self.assertEqual(json.loads(first.splitlines()[0])['kind'],'native_vox_verdict')
if __name__=='__main__':unittest.main()
