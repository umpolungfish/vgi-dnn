import unittest
from fractions import Fraction as F
from certified_execution import frame_check
from zeta_oracle import split, fuse, exact_certificate

class CertifiedTests(unittest.TestCase):
    def test_frame_all_dimensions(self):
        for d in (2,4,12):
            self.assertTrue(exact_certificate(d)['left_inverse_exact'])
            c=[F(i%7,3) for i in range(d*d)]
            self.assertEqual(fuse(split(c,d),d),c)
    def test_frame_checked_addresses(self):
        self.assertTrue(frame_check([0]*12,15))
        self.assertFalse(frame_check([0]*11+[7],15))
        self.assertFalse(frame_check([0]*12,16))
    def test_invalid_not_silently_clamped(self):
        self.assertFalse(frame_check([0]*11+[-1],0))
        self.assertFalse(frame_check([0]*12,-1))
if __name__=='__main__':unittest.main()
