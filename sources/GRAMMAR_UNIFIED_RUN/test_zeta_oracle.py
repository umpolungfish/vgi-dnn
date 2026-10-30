import unittest
from fractions import Fraction as F
from zeta_oracle import ZETA_NEG_ONE, gram_coefficients, dual_coefficients, exact_certificate, split, fuse

class ExactZetaSicTests(unittest.TestCase):
    def test_zeta_coefficient(self):
        self.assertEqual(ZETA_NEG_ONE,F(-1,12))
        self.assertEqual(dual_coefficients(2),(F(3,2),F(-1,4)))
        self.assertEqual(dual_coefficients(4),(F(5,4),F(-1,16)))
        self.assertEqual(dual_coefficients(12),(F(13,12),F(-1,144)))
    def test_exact_basis_inverses(self):
        for d in (2,4,12):
            with self.subTest(d=d):self.assertTrue(exact_certificate(d)['left_inverse_exact'])
    def test_nondiagonal_rational_coordinates(self):
        for d in (2,4,12):
            v=[F((i*17)%23-11, 7+i%5) for i in range(d*d)]
            self.assertEqual(fuse(split(v,d),d),v)
    def test_dimension_validation(self):
        with self.assertRaises(ValueError):split([F(1)],4)
        with self.assertRaises(ValueError):dual_coefficients(3)
if __name__=='__main__':unittest.main()
