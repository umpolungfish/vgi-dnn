import sqlite3, tempfile, unittest
from pathlib import Path
from grammar_corpus import index,search,decode
class CorpusTests(unittest.TestCase):
 def test_ingestion(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td);d=p/'docs';l=p/'lean';d.mkdir();l.mkdir()
   (d/'structure.md').write_text('# Grammar\nThe dual frame and carrier identity.\n')
   (l/'Test.lean').write_text('import Mathlib.Tactic\nnamespace Imscribing\ntheorem dual_frame : True := by trivial\nend Imscribing\n')
   out=index([('docs',d),('lean',l)],p/'index.sqlite')
   self.assertEqual(out['counts']['document'],1);self.assertEqual(out['counts']['lean'],1)
   self.assertEqual(out['counts']['declarations'],1);self.assertEqual(out['counts']['imports'],1)
   self.assertEqual(len(search(p/'index.sqlite','carrier',10)),1)
   with sqlite3.connect(p/'index.sqlite') as db:
    self.assertEqual(db.execute('select name from declarations').fetchone()[0],'dual_frame')
 def test_missing_root_reported(self):
  with tempfile.TemporaryDirectory() as td:
   with self.assertRaises(RuntimeError):
    index([('missing',Path(td)/'absent')],Path(td)/'a.sqlite')
 def test_docx(self):
  import zipfile
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'a.docx'
   with zipfile.ZipFile(p,'w') as z:z.writestr('word/document.xml','<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Dual frame</w:t></w:r></w:p></w:body></w:document>')
   self.assertIn('Dual frame',decode(p)[0])
if __name__=='__main__':unittest.main()

class AtomicPublicationTests(unittest.TestCase):
 def test_failed_rebuild_preserves_previous_database(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td); d=p/'docs';d.mkdir()
   (d/'a.md').write_text('unchanged source one')
   target=p/'index.sqlite'
   index([('docs',d)],target)
   with sqlite3.connect(target) as db: old=db.execute('select count(*) from files').fetchone()[0]
   with self.assertRaises(RuntimeError):index([('missing',p/'missing')],target)
   with sqlite3.connect(target) as db:self.assertEqual(db.execute('select count(*) from files').fetchone()[0],old)
 def test_rebuild_publishes_new_index(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td); d=p/'docs';d.mkdir();target=p/'index.sqlite'
   (d/'a.md').write_text('alpha first');index([('docs',d)],target)
   (d/'b.md').write_text('beta second');index([('docs',d)],target)
   with sqlite3.connect(target) as db:self.assertEqual(db.execute('select count(*) from files').fetchone()[0],2)
