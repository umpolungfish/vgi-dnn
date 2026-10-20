import unittest
import numpy as np
from sic_gate import *
class SicTests(unittest.TestCase):
 def test_complex_matrix_units(self):
  for b in BASIS:
   for v in (b,1j*b):self.assertLess(np.linalg.norm(synthesize(analyze(v))-v),1e-12)
 def test_certificate(self):
  c=certificate();self.assertLess(c['matrix_unit_reconstruction_frobenius_max'],1e-12);self.assertLess(c['second_moment_frobenius'],1e-12)
 def test_signed_urgleichung(self):
  q=urgleichung_coefficients(bloch_to_probs([0,0,0]));self.assertTrue((q>0).all())
  q=urgleichung_coefficients(bloch_to_probs([0,0,1]));self.assertTrue((q<0).any())
 def test_reject_unphysical(self):
  with self.assertRaises(ValueError):probs_to_bloch([1,0,0,0])
 def test_source_independent_four(self):
  e=FourEvidence();self.assertEqual(e.verdict('closure'),'N')
  e.put('closure','cpu','support');self.assertEqual(e.verdict('closure'),'T')
  e.put('closure','cpu','refute');self.assertEqual(e.verdict('closure'),'F')
  e.put('closure','fixed','support');self.assertEqual(e.verdict('closure'),'B')
  e.put('other','gpu','support');self.assertEqual(e.verdict('other'),'T')
 def test_no_vertex_logical_equivalence(self):
  self.assertEqual(LABELS,('N','T','F','B'))
  self.assertEqual(FourEvidence().verdict('B-coordinate-nonzero'),'N')
if __name__=='__main__':unittest.main()
