import unittest
from pathlib import Path
from primitive_dag import load_registry, WordDAG, AXES, RADICES, ALPHABET
ROOT=Path(__file__).parent
class TestPrimitiveDAG(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items=load_registry(ROOT/'SNS_PRIME.md')
        cls.dag=WordDAG(cls.items)
    def test_complete_catalog(self):
        self.assertEqual(len(self.items),49)
        self.assertEqual([sum(e['axis']==a for e in self.items) for a in AXES],list(RADICES))
    def test_exact_unfold_and_hash_consing(self):
        self.assertTrue(self.dag.report()['roundtrips_exact'])
        self.assertLess(self.dag.report()['root_nodes'],49)
    def test_original_words_are_opcode_only(self):
        self.assertTrue(all(set(e['word'])<=set(ALPHABET) for e in self.items))
    def test_full_tuple_and_wrong_axis(self):
        glyphs='𐑦𐑸𐑾𐑹𐑐𐑧𐑲𐑠⊙𐑫𐑳𐑭'
        self.assertEqual(self.dag.tuple_word(glyphs), ''.join(self.dag.entries[g]['word'] for g in glyphs))
        with self.assertRaises(ValueError):self.dag.tuple_word('𐑛'*12)
if __name__=='__main__': unittest.main()
