import json
import tempfile
import unittest
from pathlib import Path
from vox_native_evidence import probes, run

class VoxEvidenceTests(unittest.TestCase):
    def test_default_words_are_imasm_not_crystal_addresses(self):
        self.assertEqual(len(probes()), 2)
        self.assertTrue(all(word.startswith('⊢') for _,word in probes()))
    def test_tsv_validation(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'bad.tsv'; p.write_text('missing delimiter\n')
            with self.assertRaises(ValueError): probes(p)
    def test_mock_vox_preserves_stdout_stderr(self):
        with tempfile.TemporaryDirectory() as td:
            exe=Path(td)/'vox'; exe.write_text('#!/bin/sh\n[ "$1" = verdict ] && [ "$2" = --tsv ] || exit 2\ncat\necho "mock diagnostic" >&2\n'); exe.chmod(0o755)
            out=Path(td)/'report.json'
            result=run(str(exe),output=str(out))
            self.assertEqual(result['status'],'completed')
            self.assertIn('sourceWord',result['stdout'])
            self.assertIn('mock diagnostic',result['stderr'])
            self.assertEqual(json.loads(out.read_text())['input_sha256'], result['input_sha256'])
    def test_missing_vox_is_reported(self):
        with tempfile.TemporaryDirectory() as td:
            result=run('/definitely/not/a/vox', output=str(Path(td)/'out.json'))
            self.assertEqual(result['status'],'unavailable')

if __name__=='__main__': unittest.main()
