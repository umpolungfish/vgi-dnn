import json
import tempfile
import unittest
from pathlib import Path
from vox_native_evidence import probes, run

class VoxEvidenceTests(unittest.TestCase):
    def test_default_words(self):
        self.assertEqual(len(probes()), 2)
        self.assertTrue(all(word.startswith('⊢') for _, word in probes()))

    def test_tsv_validation(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'bad.tsv'
            p.write_text('missing delimiter\n')
            with self.assertRaises(ValueError):
                probes(p)

    def test_mock_vox_requires_file_argument_and_preserves_output(self):
        with tempfile.TemporaryDirectory() as td:
            exe = Path(td) / 'vox'
            exe.write_text('#!/bin/sh\n[ "$1" = verdict ] && [ "$2" = --tsv ] && [ -f "$3" ] || exit 7\ncat "$3"\necho "mock diagnostic" >&2\n')
            exe.chmod(0o755)
            out = Path(td) / 'report.json'
            result = run(str(exe), output=str(out))
            self.assertEqual(result['status'], 'completed')
            self.assertIn('sourceWord', result['stdout'])
            self.assertIn('mock diagnostic', result['stderr'])
            self.assertEqual(json.loads(out.read_text())['input_sha256'], result['input_sha256'])

    def test_cli_failure_records_real_stderr(self):
        with tempfile.TemporaryDirectory() as td:
            exe = Path(td) / 'vox'
            exe.write_text('#!/bin/sh\necho "native CLI error" >&2\nexit 5\n')
            exe.chmod(0o755)
            result = run(str(exe), output=str(Path(td) / 'out.json'))
            self.assertEqual(result['status'], 'command_failed')
            self.assertEqual(result['returncode'], 5)
            self.assertIn('native CLI error', result['stderr'])

    def test_missing_vox(self):
        with tempfile.TemporaryDirectory() as td:
            result = run('/definitely/not/a/vox', output=str(Path(td) / 'out.json'))
            self.assertEqual(result['status'], 'unavailable')

if __name__ == '__main__':
    unittest.main()
