import unittest,torch
from grammar_ops import digits_from_index,index_from_digits,apply_operation,CrystalReentry,OpExamples
from grammar_dnn import RADICES
class OperationsTests(unittest.TestCase):
    def test_codec(self):
        for n in (0,1,123,6687051,12089822,17279999): self.assertEqual(n,index_from_digits(digits_from_index(n)))
    def test_composition(self):
        old=[0]*12
        result=apply_operation(old,2,(0,1),(3,4))
        self.assertEqual([3,4]+[0]*10,result)
        self.assertEqual(old,apply_operation(old,0,(0,1),(3,4)))
    def test_dnn(self):
        model=CrystalReentry(dropout=0)
        vals=torch.tensor([[0]*14]);axes=torch.tensor([[0,1]]);mode=torch.tensor([2]);out=model(vals,axes,mode)
        self.assertEqual([o.shape for o in out],[torch.Size([1,r]) for r in RADICES])
    def test_dataset(self):
        rows=[{'targets':{'indices':[0]*12}}]
        ds=OpExamples(rows,repeat=3)
        self.assertEqual(len(ds),3)
        self.assertEqual(ds[0][0].shape,torch.Size([14]))
if __name__=='__main__':unittest.main()
