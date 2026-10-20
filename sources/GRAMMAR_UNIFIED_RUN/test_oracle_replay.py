import unittest
from oracle_replay import replay

class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.r={'case':1,'step':2,'kind':'coordinate','operation':{'mode':1,'axes':[0,1],'values':[3,2]},
                'before':{'address':[0]*12,'carrier':7},
                'proposed':{'address':[1]+[0]*11,'carrier':7},
                'oracle_target':{'address':[3]+[0]*11,'carrier':7}}
    def test_replay(self): self.assertTrue(replay(self.r))
    def test_corrupted_target_rejected(self):
        self.r['oracle_target']['address'][0]=2
        with self.assertRaises(AssertionError):replay(self.r)
    def test_nonerror_rejected(self):
        self.r['proposed']=self.r['oracle_target']
        with self.assertRaises(AssertionError):replay(self.r)
