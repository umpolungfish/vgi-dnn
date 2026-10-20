import unittest
from oracle_attribution import frame_check, apply_operation, RADICES, run

class FakeOps: pass
class FakeCarrier: pass

class AttributionTests(unittest.TestCase):
    def test_valid_wrong_coordinate_passes_frame(self):
        x=[0]*12
        wrong=x[:];wrong[0]=1
        self.assertTrue(frame_check(wrong,0))
        self.assertNotEqual(wrong,apply_operation(x,1,(0,1),(2,1)))

    def test_invalid_frame_is_rejected(self):
        self.assertFalse(frame_check([RADICES[0]]+[0]*11,0))
        self.assertFalse(frame_check([0]*12,16))

    def test_attribution_with_wrong_neural_stub(self):
        import oracle_attribution as m
        oldop,oldcarrier=m.op_step,m.carrier_step
        try:
            m.op_step=lambda model,state,a,b,u,v,mode: state[:]
            m.carrier_step=lambda model,state,arity,flags:state
            report=run(FakeOps(),FakeCarrier(),cases=4,steps=16,seed=1)
            self.assertGreater(report['oracle_correction_count'],0)
            self.assertEqual(report['counts']['invalid_frame_proposals'],0)
            self.assertEqual(report['counts']['gated_final_exact'],4)
            self.assertEqual(report['counts']['semantically_wrong_but_frame_valid'],report['oracle_correction_count'])
            for e in report['corrections']:
                self.assertTrue(e['frame_valid'])
                self.assertNotEqual(e['proposed'],e['oracle_target'])
        finally:
            m.op_step,m.carrier_step=oldop,oldcarrier

if __name__=='__main__':unittest.main()
