import unittest
from integrated_stress import assess
from grammar_ops import RADICES,apply_operation
from dynamic_bridge import evaluate

class TestIntegration(unittest.TestCase):
    def test_roundtrip(self):
        values=[r-1 for r in RADICES]
        self.assertEqual(values,apply_operation(values,0,(1,2),(0,0)))
    def test_set_two(self):
        values=[0]*12
        self.assertEqual([2,4]+[0]*10,apply_operation(values,2,(0,1),(2,4)))
    def test_carrier_oracle(self):
        for mask in range(16):
            self.assertEqual(mask,evaluate(mask,2,(0,0))['fused_mask'])
if __name__=='__main__':unittest.main()
