import unittest
from dynamic_bridge import split,reverse,evaluate,display
class TestDynamic(unittest.TestCase):
    def test_all_roundtrips(self):
        for n in range(16):
            for arity in (2,3): self.assertTrue(evaluate(n,arity,(False,)*arity)['recovered'])
    def test_arev_involution(self):
        for n in range(16):self.assertEqual(n,reverse(reverse(n)))
    def test_2arm_partition(self):
        for n in range(16):
            x,y=split(n,2);self.assertEqual(x|y,n);self.assertEqual(x&y,0)
    def test_3arm_partition(self):
        for n in range(16):
            a,b,c=split(n,3);self.assertEqual(a|b|c,n)
    def test_distinguish_structural_closure_and_flow(self):
        # AREV on truth arm changes the value while the split/fuse pairing remains shaped alike.
        self.assertFalse(evaluate(0b1001,2,(True,False))['recovered'])
    def test_both_is_not_failure(self):
        self.assertEqual(display(3),'B'); self.assertTrue(evaluate(3,2,(False,False))['recovered'])
if __name__=='__main__':unittest.main()
