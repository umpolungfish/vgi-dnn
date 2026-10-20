import unittest
from repair import generate,tensors
class TestRepair(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(generate(25,17),generate(25,17))
    def test_independent_seeds(self):
        self.assertNotEqual(generate(25,17),generate(25,18))
    def test_valid_shapes(self):
        vals,axes,mode,target=tensors(generate(6,42))
        self.assertEqual(tuple(vals.shape),(6,14))
        self.assertEqual(tuple(target.shape),(6,12))
        self.assertEqual(tuple(axes.shape),(6,2))
        self.assertEqual(tuple(mode.shape),(6,))
if __name__=='__main__':unittest.main()
