import unittest,torch
from carrier_net import CarrierKernel,masks_data,assess
class TestNeural(unittest.TestCase):
    def test_shapes(self):
        m=CarrierKernel();a,b=masks_data([0,1]);self.assertEqual(a.shape[1],5);self.assertEqual(tuple(m(a[:,:4],a[:,4:]).shape),tuple(b.shape))
    def test_all_seeds_evaluated(self):
        m=CarrierKernel();self.assertEqual(assess(m,range(16))['cases'],192)
if __name__=='__main__':unittest.main()
