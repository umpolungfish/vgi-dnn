import json
import tempfile
import unittest
from pathlib import Path
from vox_evidence_join import join
class NativeJoinTests(unittest.TestCase):
    def test_verified_join(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            p=d/'in.json'; o=d/'out.jsonl'
            p.write_text(json.dumps({'status':'completed','returncode':0,'probes':[{'name':'sourceWord','word':'⊢≻⊤'},{'name':'strandedWord','word':'⊢∋'}], 'stdout':'sourceWord\tT\t3\nstrandedWord\tN\t2\n'}))
            rows=join(p,o)
            self.assertEqual([r['verdict'] for r in rows],['T','N'])
            self.assertEqual(len(o.read_text().splitlines()),2)
    def test_incomplete_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);p=d/'in.json'
            p.write_text(json.dumps({'status':'completed','returncode':0,'probes':[{'name':'x','word':'ab'}], 'stdout':''}))
            with self.assertRaises(ValueError): join(p,d/'out')
    def test_non_native_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);p=d/'in.json'
            p.write_text(json.dumps({'status':'unavailable','returncode':None,'probes':[], 'stdout':''}))
            with self.assertRaises(ValueError): join(p,d/'out')
if __name__=='__main__': unittest.main()
