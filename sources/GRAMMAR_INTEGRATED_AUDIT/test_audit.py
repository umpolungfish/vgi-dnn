import unittest
from audit import audit_one_step, audit_rollout, predict_many, ROOT
from grammar_ops import RADICES,apply_operation

class TestAudit(unittest.TestCase):
    def test_oracle_identity(self):
        x=[0]*12
        self.assertEqual(apply_operation(x,0,(0,1),(2,3)),x)
        self.assertEqual(apply_operation(x,2,(0,1),(2,3))[:2],[2,3])
    def test_adjacent_repo(self):
        self.assertTrue((ROOT/'integrated_stress.py').is_file())
    def test_prediction_interface(self):
        from integrated_stress import load_ops
        p=ROOT/'demo_checkpoints/ops.pt'
        if not p.exists():self.skipTest('demo checkpoint absent')
        model=load_ops(p)
        row={'input':[0]*12,'a':0,'b':1,'mode':0,'v1':2,'v2':2}
        self.assertEqual(len(predict_many(model,[row])[0]),12)
if __name__=='__main__':unittest.main()
