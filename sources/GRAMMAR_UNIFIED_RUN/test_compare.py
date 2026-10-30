import unittest
from grammar_ops import RADICES, apply_operation
from certified_execution import frame_check
class ComparisonTests(unittest.TestCase):
    def test_expected_target_is_frame_valid(self):
        x=[0]*12
        y=apply_operation(x,2,(0,1),(1,2))
        self.assertEqual(len(y),len(RADICES))
        self.assertTrue(frame_check(y,0))
