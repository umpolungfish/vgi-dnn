import unittest
from grammar_dnn import GrammarDNN, make_vocab, read_records, RADICES
from pathlib import Path

class ModelContract(unittest.TestCase):
    def test_expected_default_parameter_count(self):
        rows = read_records(Path(__file__).resolve().parent / 'grammar_training/train.jsonl')
        model = GrammarDNN(len(make_vocab(rows)))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 18673)
        self.assertEqual(model.config['d'], 32)
        self.assertEqual(tuple(RADICES), (4,5,4,5,3,5,3,4,5,4,3,4))

if __name__=='__main__': unittest.main()
