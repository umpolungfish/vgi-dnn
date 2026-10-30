import json
import tempfile
import unittest
from pathlib import Path
from bootstrap import AXES
from imasm_traces import capture

class Tests(unittest.TestCase):
    def test_capture_and_resume(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            exe=root/'imasm'
            exe.write_text('#!/usr/bin/env python3\nimport sys\nprint(*sys.argv[1:])\n')
            exe.chmod(0o755)
            address=''.join(axis[2][0] for axis in AXES)
            objects=root/'objects.jsonl'
            objects.write_text(json.dumps({'name':'example','address':address,'source_line':5},ensure_ascii=False)+'\n')
            dest=root/'evidence'
            result=capture(objects,dest,str(exe),1,2,True,False,False)
            self.assertEqual(result['new_records'],51)
            last=json.loads((dest/'traces.jsonl').read_text().splitlines()[-1])
            self.assertEqual(list(last['commands']),['check','eval','eval16'])
            self.assertIsNone(last['flow_judgment'])
            self.assertEqual(capture(objects,dest,str(exe),1,2,True,False,True)['new_records'],0)
    def test_missing_executable(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'objects.jsonl';p.write_text('')
            with self.assertRaises(FileNotFoundError):
                capture(p,Path(t)/'out','/does/not/exist')
if __name__=='__main__': unittest.main()
