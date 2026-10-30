import unittest
from pathlib import Path
class TestComparison(unittest.TestCase):
    def test_script_and_dependencies(self):
        root=Path(__file__).resolve().parent.parent
        self.assertTrue((root/'GRAMMAR_INTEGRATED_V1'/'integrated_stress.py').is_file())
        self.assertTrue((Path(__file__).parent/'compare.py').is_file())
    def test_assessment_is_seed_deterministic(self):
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'GRAMMAR_INTEGRATED_V1'))
        from integrated_stress import assess
        from grammar_ops import CrystalReentry
        from carrier_net import CarrierKernel
        model=CrystalReentry(); carrier=CarrierKernel()
        import torch
        torch.set_num_threads(1)
        self.assertEqual(assess(model,carrier,2,2,seed=21),assess(model,carrier,2,2,seed=21))
if __name__=='__main__':unittest.main()
